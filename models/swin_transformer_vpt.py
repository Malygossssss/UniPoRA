#!/usr/bin/env python3
"""
swin transformer with prompt
"""
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn import Conv2d, Dropout

from timm.models.layers import to_2tuple

from .swin_transformer_mtlora import (
    SwinTransformerMTLoRA,
    SwinTransformerBlock,
    PatchMerging,
    window_partition,
    window_reverse,
    WindowAttention,
)

logger = logging.getLogger("visual_prompt")


class PromptedSwinTransformer(SwinTransformerMTLoRA):
    def __init__(
            self,
            prompt_config,
            tasks,
            mtlora,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            **kwargs,
    ):
        if prompt_config.LOCATION == "pad":
            img_size += 2 * prompt_config.NUM_TOKENS

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            fused_window_process=kwargs.get("fused_window_process", False),
            tasks=tasks,
            mtlora=mtlora,
        )

        # rebuild layers with prompt-aware blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PromptedBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PromptedPatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                num_prompts=prompt_config.NUM_TOKENS,
                prompt_location=prompt_config.LOCATION,
                deep_prompt=prompt_config.DEEP,
                tasks=tasks,
                mtlora=mtlora,
                layer_idx=i_layer,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.prompt_config = prompt_config
        self.tasks = tasks
        self.mtlora = mtlora
        self.share_task_prompt = getattr(self.prompt_config, "SHARE_TASK_PROMPT", False)
        self.prompt_keys = ["shared"] if self.share_task_prompt else list(tasks)
        self.task_to_idx = {task: idx for idx, task in enumerate(tasks)}
        self.num_base_prompts = len(self.prompt_keys)
        self.num_tasks = len(tasks)
        self.use_dynamic_prompts = False
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        if self.prompt_config.LOCATION == "add":
            num_tokens = self.embeddings.position_embeddings.shape[1]
        elif self.prompt_config.LOCATION == "add-1":
            num_tokens = 1
        else:
            num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens
        pool_size_cfg = getattr(self.prompt_config, "DEEP_PROMPT_POOL_SIZE", 0)
        default_pool_size = (
            pool_size_cfg if pool_size_cfg and pool_size_cfg > 0 else self.num_tokens
        )
        stage_pool_sizes_cfg = getattr(self.prompt_config, "DEEP_PROMPT_POOL_SIZES", None)
        if stage_pool_sizes_cfg:
            if len(stage_pool_sizes_cfg) != self.num_layers:
                raise ValueError(
                    "DEEP_PROMPT_POOL_SIZES must have one entry per Swin stage"
                )
            self.prompt_pool_sizes = []
            for size in stage_pool_sizes_cfg:
                stage_size = int(size) if size is not None else 0
                if stage_size <= 0:
                    stage_size = default_pool_size
                self.prompt_pool_sizes.append(stage_size)
        else:
            self.prompt_pool_sizes = [default_pool_size for _ in range(self.num_layers)]
        self.prompt_pool_size = default_pool_size

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
        if self.prompt_config.PROJECT > -1:
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, embed_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
        else:
            self.prompt_proj = nn.Identity()

        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + embed_dim))

            if self.prompt_config.LOCATION == "below":
                self.patch_embed.proj = Conv2d(
                    in_channels=num_tokens + 3,
                    out_channels=embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
                nn.init.zeros_(self.patch_embed.proj.bias)

                self.prompt_embeddings = nn.ParameterDict()
                if self.share_task_prompt:
                    emb = torch.zeros(1, num_tokens, img_size[0], img_size[1])
                    nn.init.uniform_(emb, -val, val)
                    self.prompt_embeddings["shared"] = nn.Parameter(emb)
                else:
                    for task in tasks:
                        emb = torch.zeros(1, num_tokens, img_size[0], img_size[1])
                        nn.init.uniform_(emb, -val, val)
                        self.prompt_embeddings[task] = nn.Parameter(emb)

            elif self.prompt_config.LOCATION == "pad":
                self.prompt_embeddings_tb = nn.ParameterDict()
                self.prompt_embeddings_lr = nn.ParameterDict()
                if self.share_task_prompt:
                    emb_tb = torch.zeros(1, 3, 2 * num_tokens, img_size[0])
                    emb_lr = torch.zeros(1, 3, img_size[0] - 2 * num_tokens, 2 * num_tokens)
                    nn.init.uniform_(emb_tb, 0.0, 1.0)
                    nn.init.uniform_(emb_lr, 0.0, 1.0)
                    self.prompt_embeddings_tb["shared"] = nn.Parameter(emb_tb)
                    self.prompt_embeddings_lr["shared"] = nn.Parameter(emb_lr)
                else:
                    for task in tasks:
                        emb_tb = torch.zeros(1, 3, 2 * num_tokens, img_size[0])
                        emb_lr = torch.zeros(1, 3, img_size[0] - 2 * num_tokens, 2 * num_tokens)
                        nn.init.uniform_(emb_tb, 0.0, 1.0)
                        nn.init.uniform_(emb_lr, 0.0, 1.0)
                        self.prompt_embeddings_tb[task] = nn.Parameter(emb_tb)
                        self.prompt_embeddings_lr[task] = nn.Parameter(emb_lr)

                self.prompt_norm = tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )

            else:
                self.prompt_embeddings = nn.ParameterDict()
                if self.share_task_prompt:
                    emb = torch.zeros(1, num_tokens, embed_dim)
                    nn.init.uniform_(emb, -val, val)
                    self.prompt_embeddings["shared"] = nn.Parameter(emb)
                else:
                    for task in tasks:
                        emb = torch.zeros(1, num_tokens, embed_dim)
                        nn.init.uniform_(emb, -val, val)
                        self.prompt_embeddings[task] = nn.Parameter(emb)

                if self.prompt_config.DEEP:
                    self.use_dynamic_prompts = getattr(
                        self.prompt_config, "DYNAMIC_PROMPT", False
                    )
                    if self.use_dynamic_prompts and self.share_task_prompt:
                        raise ValueError(
                            "Dynamic prompts require individual task prompts."
                        )

                    if self.use_dynamic_prompts:
                        self.deep_prompt_pools = nn.ParameterList()
                        self.deep_prompt_gates = nn.ParameterList()
                        for layer_idx, depth in enumerate(depths):
                            stage_dim = embed_dim * (2 ** layer_idx)
                            stage_pool_size = self.prompt_pool_sizes[layer_idx]

                            if (
                                self.prompt_config.LOCATION == "prepend"
                                and layer_idx == 0
                            ):
                                num_prompt_blocks = max(depth - 1, 0)
                            else:
                                num_prompt_blocks = depth

                            pool_shape = (
                                num_prompt_blocks,
                                stage_pool_size,
                                stage_dim,
                            )
                            prompt_pool = torch.zeros(pool_shape)
                            if prompt_pool.numel() > 0:
                                nn.init.uniform_(prompt_pool, -val, val)
                            self.deep_prompt_pools.append(nn.Parameter(prompt_pool))

                            gate_shape = (
                                num_prompt_blocks,
                                self.num_tasks,
                                num_tokens,
                                stage_pool_size,
                            )
                            gate_param = torch.zeros(gate_shape)
                            if (
                                num_prompt_blocks > 0
                                and stage_pool_size > 0
                                and num_tokens > 0
                            ):
                                base = torch.zeros(
                                    self.num_tasks, num_tokens, stage_pool_size
                                )
                                for task_idx in range(self.num_tasks):
                                    for token_idx in range(num_tokens):
                                        pool_idx = (token_idx + task_idx) % stage_pool_size
                                        base[task_idx, token_idx, pool_idx] = 1.0
                                base = base.unsqueeze(0)
                                gate_param.copy_(
                                    base.repeat(num_prompt_blocks, 1, 1, 1)
                                )
                            self.deep_prompt_gates.append(nn.Parameter(gate_param))
                    else:
                        self.deep_prompt_embeddings = nn.ModuleList()
                        for layer_idx, depth in enumerate(depths):
                            stage_dim = embed_dim * (2 ** layer_idx)
                            if (
                                self.prompt_config.LOCATION == "prepend"
                                and layer_idx == 0
                            ):
                                num_prompt_blocks = max(depth - 1, 0)
                            else:
                                num_prompt_blocks = depth

                            prompt_dict = nn.ParameterDict()
                            if self.share_task_prompt:
                                emb = torch.zeros(
                                    num_prompt_blocks, num_tokens, stage_dim
                                )
                                nn.init.uniform_(emb, -val, val)
                                prompt_dict["shared"] = nn.Parameter(emb)
                            else:
                                for task in tasks:
                                    emb = torch.zeros(
                                        num_prompt_blocks, num_tokens, stage_dim
                                    )
                                    nn.init.uniform_(emb, -val, val)
                                    prompt_dict[task] = nn.Parameter(emb)
                            self.deep_prompt_embeddings.append(prompt_dict)
        else:
            raise ValueError("Other initiation scheme is not supported")

        self._compact_prompt_layout = False
        self.reset_prompt_pruning()
        self.clear_prompt_runtime_gates()

    def forward(self, x, task=None, return_stages=False, flatten_ft=False):
        x = self.forward_features(x, task, return_stages)
        if return_stages:
            return x
        x = self.head(x)
        if flatten_ft:
            return x
        return x

    def incorporate_prompt(self, x, task):
        """Combine prompt embeddings with image-patch embeddings for a task."""
        B = x.shape[0]

        if self.prompt_config.LOCATION == "prepend":
            x = self.get_patch_embeddings(x)
            prompt_embd = self.prompt_dropout(
                self._select_layer_prompt(
                    self._select_prompt(self.prompt_embeddings, task),
                    task,
                    layer_idx=0,
                ).expand(B, -1, -1)
            )
            x = torch.cat((prompt_embd, x), dim=1)

        elif self.prompt_config.LOCATION == "add":
            x = self.get_patch_embeddings(x)
            x = x + self.prompt_dropout(
                self._select_prompt(self.prompt_embeddings, task).expand(B, -1, -1)
            )

        elif self.prompt_config.LOCATION == "add-1":
            x = self.get_patch_embeddings(x)
            L = x.shape[1]
            prompt_emb = self.prompt_dropout(
                self._select_prompt(self.prompt_embeddings, task).expand(B, -1, -1)
            )
            x = x + prompt_emb.expand(-1, L, -1)

        elif self.prompt_config.LOCATION == "pad":
            prompt_emb_lr = self.prompt_norm(
                self._select_prompt(self.prompt_embeddings_lr, task)
            ).expand(B, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(
                self._select_prompt(self.prompt_embeddings_tb, task)
            ).expand(B, -1, -1, -1)
            x = torch.cat(
                (
                    prompt_emb_lr[:, :, :, : self.num_tokens],
                    x,
                    prompt_emb_lr[:, :, :, self.num_tokens :],
                ),
                dim=-1,
            )
            x = torch.cat(
                (
                    prompt_emb_tb[:, :, : self.num_tokens, :],
                    x,
                    prompt_emb_tb[:, :, self.num_tokens :, :],
                ),
                dim=-2,
            )
            x = self.get_patch_embeddings(x)

        elif self.prompt_config.LOCATION == "below":
            x = torch.cat(
                (
                    x,
                    self.prompt_norm(self._select_prompt(self.prompt_embeddings, task)).expand(
                        B, -1, -1, -1
                    ),
                ),
                dim=1,
            )
            x = self.get_patch_embeddings(x)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x

    def _compose_dynamic_prompt(self, layer_idx, task):
        pool = self.deep_prompt_pools[layer_idx]
        gates = self.deep_prompt_gates[layer_idx]

        num_prompt_blocks = gates.shape[0]
        stage_dim = pool.shape[-1] if pool.ndim == 3 else 0

        if num_prompt_blocks == 0 or pool.shape[1] == 0:
            return pool.new_zeros((num_prompt_blocks, self.num_tokens, stage_dim))

        task_idx = self.task_to_idx[task]
        weights = F.softmax(gates[:, task_idx, :, :], dim=-1)
        composed = torch.einsum("bts,bsd->btd", weights, pool)
        return self._select_layer_prompt(composed, task, layer_idx)

    def get_patch_embeddings(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def _select_prompt(self, prompt_dict, task):
        key = "shared" if self.share_task_prompt else task
        return prompt_dict[key]

    def _resolve_prompt_key(self, task):
        return "shared" if self.share_task_prompt else task

    def _get_prompt_token_count(self, layer_idx, task):
        key = self._resolve_prompt_key(task)
        if self.prompt_config.LOCATION == "pad":
            return int(self.num_tokens)
        if layer_idx == 0:
            return int(self._select_prompt(self.prompt_embeddings, key).shape[1])
        if not getattr(self.prompt_config, "DEEP", False):
            return int(self._select_prompt(self.prompt_embeddings, key).shape[1]) if hasattr(self, "prompt_embeddings") else int(self.num_tokens)
        if getattr(self, "use_dynamic_prompts", False):
            return int(self.deep_prompt_gates[layer_idx].shape[2])
        return int(self._select_prompt(self.deep_prompt_embeddings[layer_idx], key).shape[1])

    def _get_stage_prompt_count(self, task, layer_idx):
        if getattr(self, "_compact_prompt_layout", False):
            return self._get_prompt_token_count(layer_idx, task)
        key = self._resolve_prompt_key(task)
        layer_indices = self._prompt_active_indices.get(key, {})
        if layer_idx not in layer_indices:
            return 0
        return int(layer_indices[layer_idx].numel())

    def enable_compact_prompt_layout(self, enabled=True):
        self._compact_prompt_layout = bool(enabled)
        if self._compact_prompt_layout:
            self.reset_prompt_pruning()

    def reset_prompt_pruning(self):
        self._prompt_active_indices = {}
        for key in self.prompt_keys:
            self._prompt_active_indices[key] = {}
            for layer_idx in range(self.num_layers):
                token_count = self._get_prompt_token_count(layer_idx, key)
                self._prompt_active_indices[key][layer_idx] = torch.arange(
                    token_count, dtype=torch.long
                )

    def set_prompt_pruning(self, keep_indices):
        self._compact_prompt_layout = False
        updated = {}
        for key in self.prompt_keys:
            source = keep_indices.get(key, keep_indices.get("shared", {}))
            updated[key] = {}
            for layer_idx in range(self.num_layers):
                default_indices = torch.arange(
                    self._get_prompt_token_count(layer_idx, key), dtype=torch.long
                )
                indices = source.get(layer_idx, source.get(str(layer_idx), default_indices))
                if isinstance(indices, torch.Tensor):
                    tensor_indices = indices.detach().cpu().long()
                else:
                    tensor_indices = torch.tensor(indices, dtype=torch.long)
                updated[key][layer_idx] = tensor_indices
        self._prompt_active_indices = updated

    def export_prompt_pruning(self):
        return {
            key: {
                int(layer_idx): indices.detach().cpu().tolist()
                for layer_idx, indices in layer_map.items()
            }
            for key, layer_map in self._prompt_active_indices.items()
        }

    def clear_prompt_runtime_gates(self):
        self._prompt_runtime_gates = None

    def set_prompt_runtime_gates(self, gates):
        self._prompt_runtime_gates = gates

    def build_prompt_runtime_gates(self, device=None, requires_grad=True):
        if device is None:
            device = next(self.parameters()).device
        runtime_gates = {}
        for key in self.prompt_keys:
            runtime_gates[key] = {}
            for layer_idx in range(self.num_layers):
                token_count = self._get_prompt_token_count(layer_idx, key)
                gate = torch.ones(
                    token_count,
                    device=device,
                    dtype=next(self.parameters()).dtype,
                    requires_grad=requires_grad,
                )
                runtime_gates[key][layer_idx] = gate
        return runtime_gates

    def _get_layer_indices(self, task, layer_idx, device):
        if getattr(self, "_compact_prompt_layout", False):
            return None
        key = self._resolve_prompt_key(task)
        layer_indices = self._prompt_active_indices.get(key, {})
        if layer_idx not in layer_indices:
            return None
        return layer_indices[layer_idx].to(device)

    def _get_layer_gate(self, task, layer_idx, device, indices=None):
        if self._prompt_runtime_gates is None:
            return None
        key = self._resolve_prompt_key(task)
        gate_map = self._prompt_runtime_gates.get(key)
        if gate_map is None and self.share_task_prompt:
            gate_map = self._prompt_runtime_gates.get("shared")
        if gate_map is None or layer_idx not in gate_map:
            return None
        gate = gate_map[layer_idx].to(device)
        if indices is not None:
            gate = gate.index_select(0, indices.to(gate.device))
        return gate

    def _select_layer_prompt(self, prompt_tensor, task, layer_idx):
        indices = self._get_layer_indices(task, layer_idx, prompt_tensor.device)
        if indices is not None:
            prompt_tensor = prompt_tensor.index_select(1, indices)
        gate = self._get_layer_gate(task, layer_idx, prompt_tensor.device, indices=indices)
        if gate is not None:
            prompt_tensor = prompt_tensor * gate.view(1, -1, 1)
        return prompt_tensor

    def train(self, mode=True):
        if mode and not getattr(self.mtlora, "ENABLED", False):
            for module in self.children():
                module.train(False)
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x, task, return_stages=False):
        x = self.incorporate_prompt(x, task)

        feats = []
        if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
            for layer_idx, layer in enumerate(self.layers):
                if getattr(self, "use_dynamic_prompts", False):
                    deep_prompt_embd = self.prompt_dropout(
                        self._compose_dynamic_prompt(layer_idx, task)
                    )
                else:
                    deep_prompt_embd = self.prompt_dropout(
                        self._select_layer_prompt(
                            self._select_prompt(
                                self.deep_prompt_embeddings[layer_idx], task
                            ),
                            task,
                            layer_idx,
                        )
                    )
                x = layer(x, task, deep_prompt_embd)
                if return_stages:
                    prompt_count = self._get_stage_prompt_count(task, layer_idx)
                    feats.append(x[:, prompt_count:, :])
        else:
            for layer_idx, layer in enumerate(self.layers):
                x = layer(x, task)
                if return_stages:
                    if self.prompt_config.LOCATION == "prepend":
                        prompt_count = self._get_stage_prompt_count(
                            task,
                            layer_idx if self.prompt_config.DEEP else 0,
                        )
                        feats.append(x[:, prompt_count:, :])
                    else:
                        feats.append(x)
        if self.norm is not None:
            x = self.norm(x)
        if return_stages:
            return feats
        if self.prompt_config.LOCATION == "prepend":
            last_layer_idx = self.num_layers - 1 if self.prompt_config.DEEP else 0
            x = x[:, self._get_stage_prompt_count(task, last_layer_idx):, :]
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def load_state_dict(self, state_dict, strict):
        if self.prompt_config.LOCATION == "below":
            # modify state_dict first   [768, 4, 16, 16]
            conv_weight = state_dict["patch_embed.proj.weight"]
            conv_weight = torch.cat(
                (conv_weight, self.patch_embed.proj.weight[:, 3:, :, :]),
                dim=1
            )
            state_dict["patch_embed.proj.weight"] = conv_weight

        super(PromptedSwinTransformer, self).load_state_dict(state_dict, strict)


class PromptedBasicLayer(nn.Module):
    def __init__(
            self,
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            num_prompts=None,
            prompt_location=None,
            deep_prompt=None,
            tasks=None,
            mtlora=None,
            layer_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.tasks = tasks
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        self.deep_prompt = deep_prompt

        self.blocks = nn.ModuleList([
            PromptedSwinTransformerBlock(
                num_prompts,
                prompt_location,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                tasks=tasks,
                mtlora=mtlora,
                layer_idx=layer_idx,
                lora=(i == depth - 1),
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(
                num_prompts,
                prompt_location,
                deep_prompt,
                input_resolution,
                dim=dim,
                norm_layer=norm_layer,
                layer_idx=layer_idx,
                mtlora=mtlora,
            )
        else:
            self.downsample = None

    def forward(self, x, task, deep_prompt_embd=None):
        if self.deep_prompt and deep_prompt_embd is None and self.prompt_location == "prepend":
            raise ValueError("need deep_prompt embeddings")

        prompt_count = x.shape[1] - (self.input_resolution[0] * self.input_resolution[1]) \
            if self.prompt_location == "prepend" else 0

        if not self.deep_prompt:
            for blk in self.blocks:
                x, tasks_lora = blk(x)
                if tasks_lora is not None:
                    x = tasks_lora[task]
        else:
            B = x.shape[0]
            num_blocks = len(self.blocks)
            if deep_prompt_embd.shape[0] != num_blocks:
                for i in range(num_blocks):
                    if i == 0:
                        x, tasks_lora = self.blocks[i](x)
                        if tasks_lora is not None:
                            x = tasks_lora[task]
                    else:
                        current_prompt_count = (
                            x.shape[1] - (self.input_resolution[0] * self.input_resolution[1])
                            if self.prompt_location == "prepend"
                            else 0
                        )
                        prompt_emb = deep_prompt_embd[i - 1].expand(B, -1, -1)
                        x = torch.cat((prompt_emb, x[:, current_prompt_count:, :]), dim=1)
                        x, tasks_lora = self.blocks[i](x)
                        if tasks_lora is not None:
                            x = tasks_lora[task]
            else:
                for i in range(num_blocks):
                    current_prompt_count = (
                        x.shape[1] - (self.input_resolution[0] * self.input_resolution[1])
                        if self.prompt_location == "prepend"
                        else 0
                    )
                    prompt_emb = deep_prompt_embd[i].expand(B, -1, -1)
                    x = torch.cat((prompt_emb, x[:, current_prompt_count:, :]), dim=1)
                    x, tasks_lora = self.blocks[i](x)
                    if tasks_lora is not None:
                        x = tasks_lora[task]

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PromptedPatchMerging(PatchMerging):
    r""" Patch Merging Layer."""

    def __init__(
            self,
            num_prompts,
            prompt_location,
            deep_prompt,
            input_resolution,
            dim,
            norm_layer=nn.LayerNorm,
            layer_idx=0,
            mtlora=None,
    ):
        super().__init__(input_resolution, dim, norm_layer, layer_idx=layer_idx, mtlora=mtlora)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if prompt_location == "prepend":
            if not deep_prompt:
                self.prompt_upsampling = None
            else:
                self.prompt_upsampling = None

    def upsample_prompt(self, prompt_emb):
        if self.prompt_upsampling is not None:
            prompt_emb = self.prompt_upsampling(prompt_emb)
        else:
            prompt_emb = torch.cat(
                (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        prompt_count = L - (H * W) if self.prompt_location == "prepend" else 0

        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, :prompt_count, :]
            x = x[:, prompt_count:, :]
            L = L - prompt_count
            prompt_emb = self.upsample_prompt(prompt_emb)

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.norm(x)
        x, _ = self.reduction(x)

        return x


class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
            self,
            num_prompts,
            prompt_location,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            tasks=None,
            mtlora=None,
            layer_idx=0,
            lora=False,
    ):
        super().__init__(
            dim,
            input_resolution,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
            fused_window_process=False,
            lora=lora,
            tasks=tasks,
            mtlora=mtlora,
            layer_idx=layer_idx,
        )
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.prompt_location == "prepend":
            self.attn = PromptedWindowAttention(
                num_prompts,
                prompt_location,
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                tasks=tasks,
                mtlora=mtlora,
                layer_idx=layer_idx,
                lora=lora,
            )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        prompt_count = L - (H * W) if self.prompt_location == "prepend" else 0

        if self.prompt_location == "prepend":
            prompt_emb = x[:, :prompt_count, :]
            x = x[:, prompt_count:, :]
            L = L - prompt_count

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H * W, L)

        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        num_windows = int(x_windows.shape[0] / B)
        if self.prompt_location == "prepend":
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, prompt_count, C))
            x_windows = torch.cat((prompt_emb, x_windows), dim=1)

        attn_windows, attn_windows_lora_tasks = self.attn(x_windows, mask=self.attn_mask)

        prompt_emb_lora_tasks = {}
        if self.prompt_location == "prepend":
            prompt_emb = attn_windows[:, :prompt_count, :]
            attn_windows = attn_windows[:, prompt_count:, :]
            prompt_emb = prompt_emb.view(-1, B, prompt_count, C)
            prompt_emb = prompt_emb.mean(0)
            if attn_windows_lora_tasks is not None:
                for task in self.tasks:
                    prompt_emb_lora_tasks[task] = attn_windows_lora_tasks[task][:, :prompt_count, :]
                    attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task][:, prompt_count:, :]
                    prompt_emb_lora_tasks[task] = prompt_emb_lora_tasks[task].view(-1, B, prompt_count, C)
                    prompt_emb_lora_tasks[task] = prompt_emb_lora_tasks[task].mean(0)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        if attn_windows_lora_tasks is not None:
            for task in self.tasks:
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task].view(
                    -1, self.window_size, self.window_size, C
                )
                attn_windows_lora_tasks[task] = window_reverse(
                    attn_windows_lora_tasks[task], self.window_size, H, W
                )
                if self.shift_size > 0:
                    attn_windows_lora_tasks[task] = torch.roll(
                        attn_windows_lora_tasks[task], shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                    )
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task].view(B, H * W, C)
                if self.prompt_location == "prepend":
                    attn_windows_lora_tasks[task] = torch.cat(
                        (prompt_emb_lora_tasks[task], attn_windows_lora_tasks[task]), dim=1
                    )
                attn_windows_lora_tasks[task] = shortcut + self.drop_path(attn_windows_lora_tasks[task])

        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)
        x = shortcut + self.drop_path(x)

        mlp_input_tasks = (
            {task: self.norm2(attn_windows_lora_tasks[task]) for task in self.tasks}
            if attn_windows_lora_tasks is not None
            else None
        )
        mlp_result, mlp_lora_tasks = self.mlp(self.norm2(x), mlp_input_tasks)

        if mlp_lora_tasks is None:
            return x + self.drop_path(mlp_result), None
        else:
            if attn_windows_lora_tasks is None:
                for task in self.tasks:
                    mlp_lora_tasks[task] = self.drop_path(mlp_lora_tasks[task])
            else:
                for task in self.tasks:
                    mlp_lora_tasks[task] = attn_windows_lora_tasks[task] + self.drop_path(mlp_lora_tasks[task])
            return x + self.drop_path(mlp_result), mlp_lora_tasks


class PromptedWindowAttention(WindowAttention):
    def __init__(
            self,
            num_prompts,
            prompt_location,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            tasks=None,
            mtlora=None,
            layer_idx=0,
            lora=False,
    ):
        super().__init__(
            dim,
            window_size,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            lora=lora,
            tasks=tasks,
            mtlora=mtlora,
            layer_idx=layer_idx,
        )
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        prompt_count = (
            N - (self.window_size[0] * self.window_size[1])
            if self.prompt_location == "prepend"
            else 0
        )
        qkv, _ = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        if self.prompt_location == "prepend" and prompt_count > 0:
            _C, _H, _W = relative_position_bias.shape
            relative_position_bias = torch.cat(
                (torch.zeros(_C, prompt_count, _W, device=attn.device), relative_position_bias), dim=1
            )
            relative_position_bias = torch.cat(
                (torch.zeros(_C, _H + prompt_count, prompt_count, device=attn.device), relative_position_bias), dim=-1
            )

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            if self.prompt_location == "prepend" and prompt_count > 0:
                mask = torch.cat(
                    (torch.zeros(nW, prompt_count, _W, device=attn.device), mask), dim=1
                )
            if self.prompt_location == "prepend" and prompt_count > 0:
                mask = torch.cat(
                    (
                        torch.zeros(
                            nW, _H + prompt_count, prompt_count, device=attn.device
                        ),
                        mask,
                    ),
                    dim=-1,
                )
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x, x_proj_lora_tasks = self.proj(x)
        x = self.proj_drop(x)
        if x_proj_lora_tasks is not None:
            for task in self.tasks:
                x_proj_lora_tasks[task] = self.proj_drop(x_proj_lora_tasks[task])
        return x, x_proj_lora_tasks
