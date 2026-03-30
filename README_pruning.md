# UniPoRA TA-Prompt Token Pruning

## Scope

This pipeline only prunes TA-Prompt tokens. It does not prune layers, GA-LoRA weights, or the backbone. Pruning is always done per task and per stage: each stage sorts its own prompt tokens and removes the lowest-scoring fraction while keeping at least `MIN_TOKENS_PER_LAYER`.

The implementation is driven by `run_unipora_pruning.py` plus the stage YAMLs under `configs/mtlora/tiny_448/pascal/`.

## Start From A Trained UniPoRA Checkpoint

Use the same dataset path and task list that were used to train the checkpoint. The pruning script reads the trained checkpoint from `--resume` and treats it as the full teacher checkpoint unless `PRUNING.RECOVERY.TEACHER_CHECKPOINT` is overridden.

Example:

```bash
python run_unipora_pruning.py \
  --cfg configs/mtlora/tiny_448/pascal/unipora_prune_stage1_base_p10.yaml \
  --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 12 \
  --resume /path/to/unipora_teacher_checkpoint.pth
```

## Recommended Run Order

Stage 1 first:

1. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_base_p5.yaml`
2. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_base_p10.yaml`
3. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_base_p15.yaml`

For a direct no-recovery comparison between base importance and the stage-3 TA-replacement-aware importance, run:

1. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_tarepl_p5.yaml`
2. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_tarepl_p10.yaml`
3. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_tarepl_p15.yaml`

For post-training task-aware greedy pruning on a fixed search-val split from train, run:

1. `configs/mtlora/tiny_448/pascal/unipora_greedy_base.yaml`
2. `configs/mtlora/tiny_448/pascal/unipora_greedy_ga.yaml`

If stage 1 shows that around 10% pruning is stable, run stage 2:

1. `configs/mtlora/tiny_448/pascal/unipora_prune_stage2_base_recover_p10.yaml`
2. `configs/mtlora/tiny_448/pascal/unipora_prune_stage2_base_recover_p20.yaml`
3. `configs/mtlora/tiny_448/pascal/unipora_prune_stage2_base_recover_p30.yaml`

Then run stage 3 for a fair comparison against stage 2:

1. `configs/mtlora/tiny_448/pascal/unipora_prune_stage3_ga_recover_p10.yaml`
2. `configs/mtlora/tiny_448/pascal/unipora_prune_stage3_ga_recover_p20.yaml`
3. `configs/mtlora/tiny_448/pascal/unipora_prune_stage3_ga_recover_p30.yaml`

## What Each Stage Does

Stage 1:

- computes `S_base`
- applies per-layer prompt pruning
- evaluates immediately without recovery

Stage 2:

- computes `S_base`
- applies per-layer prompt pruning
- runs 5 recovery epochs by default
- freezes backbone and GA-LoRA by default
- trains kept TA-Prompt plus task heads

Stage 3:

- computes `S_final = S_base * exp(-beta * relu(tilde_Delta_off - tilde_Delta_on))`
- keeps the same pruning and recovery recipe as stage 2
- changes only the importance score

Stage 1 TA-replacement comparison configs reuse the same `S_final` score as stage 3, but disable recovery so that
the score itself can be compared directly against the base stage-1 runs.

Greedy post-training pruning:

- uses a fixed `search-val` split carved from train only
- evaluates accept/reject on the current task only
- searches task by task, then layer by layer
- applies step sizes `10% -> 5% -> 2% -> 1 token`
- never runs recovery or fine-tuning
- supports both `base` and `ga` importance
- uses `search-val` only for greedy decisions, then runs one final evaluation on the normal training-time `val` split after the mask is fixed
- supports greedy accept/reject for:
  - `semseg / human_parts / sal` via `mIoU` (higher is better)
  - `normals` via `mean` (lower is better)
  - `depth` via `rmse` (lower is better)
  - `edge` via `odsF` when available, otherwise `loss`

Where:

- `S_base(t,l,k) = |dL_t / dm[t,l,k]|`
- `Delta_on(t,l,k) = L01(t,l,k) - L11(t,l,k)`
- `Delta_off(t,l,k) = L00(t,l,k) - L10(t,l)`
- `tilde_Delta_on/off` are stage-wise normalized positive deltas
- `R(t,l,k) = relu(tilde_Delta_off(t,l,k) - tilde_Delta_on(t,l,k))`
- `beta` and `eps` are controlled by `PRUNING.IMPORTANCE.STAGE3_IMPORTANCE_BETA` and `PRUNING.IMPORTANCE.STAGE3_IMPORTANCE_EPS`

## Output Layout

Each run writes to:

`output/<model_name>/<tag>/pruning/<experiment_name>/`

The directory contains:

- `importance_scores.pth`
- `importance_scores.json`
- `pruning_mask.pth`
- `pruning_mask.json`
- `pruning_summary.json`
- `pruned_checkpoint.pth`
- `eval_before_recovery.json`
- `recovery_checkpoint.pth`
- `eval_after_recovery.json`
- `experiment_summary.json`

Greedy runs additionally write:

- `search_val_split.json`
- `initial_search_val_metrics.json`
- `greedy_trials.jsonl`
- `greedy_summary.json`
- `final_pruning_mask.pth`
- `final_pruning_mask.json`
- `final_pruned_checkpoint.pth`
- `final_search_val_metrics.json`
- `final_test_metrics.json`
- `final_val_metrics.json`
- `final_prompt_statistics.json`
- `importance_snapshots/`

If you want to re-evaluate a saved pruned checkpoint later, use the dedicated standalone evaluator:

```bash
python run_unipora_pruned_eval.py \
  --cfg configs/mtlora/tiny_448/pascal/unipora_greedy_base.yaml \
  --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 12 \
  --checkpoint /path/to/final_pruned_checkpoint.pth
```

This standalone evaluator defaults to `--split val`, which matches the repository's normal validation logic.

`pruning_summary.json` now stores the full pruning payload, including:

- `score_type`
- `prune_ratio`
- `min_tokens_per_layer`
- `keep_indices`
- `summary`
- `prompt_statistics`
- `aggregate`

The `aggregate` section includes:

- `total_original_tokens`
- `total_kept_tokens`
- `total_pruned_tokens`
- `overall_keep_ratio`
- `overall_prune_ratio`

## Task-Wise Mixed Pruning

You can now mix pruning ratios across tasks in a single run with:

- `PRUNING.PRUNER.RATIO`
- `PRUNING.PRUNER.TASK_RATIOS`

If `TASK_RATIOS` is provided, each listed task uses its own ratio and any task not listed falls back to the
global `RATIO`. Setting a task ratio to `0.0` keeps that task unpruned.

Selective recovery is also supported with:

- `PRUNING.RECOVERY.PROMPT_TASKS`
- `PRUNING.RECOVERY.HEAD_TASKS`
- `PRUNING.RECOVERY.LOSS_TASKS`
- `PRUNING.RECOVERY.DISTILL_TASKS`

Empty lists preserve the old behavior and default to all tasks. If you provide a subset, only those tasks
participate in the corresponding recovery component.

Selective prompt recovery is implemented against task-specific prompt parameters. If the backbone is using
dynamic prompts, the current code requires selecting all tasks together and will raise an explicit error for
partial prompt-task subsets.

Example mixed-task configs:

- `configs/mtlora/tiny_448/pascal/unipora_prune_taskmix_base_maskonly_sem30_sal20_human15_norm0.yaml`
- `configs/mtlora/tiny_448/pascal/unipora_prune_taskmix_base_selective_recover_sem30_sal20_human15_norm0.yaml`

## Greedy Commands

Base importance:

```bash
python run_unipora_greedy_pruning.py \
  --cfg configs/mtlora/tiny_448/pascal/unipora_greedy_base.yaml \
  --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 12 \
  --resume /path/to/unipora_teacher_checkpoint.pth
```

TA-replacement-aware GA importance:

```bash
python run_unipora_greedy_pruning.py \
  --cfg configs/mtlora/tiny_448/pascal/unipora_greedy_ga.yaml \
  --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 12 \
  --resume /path/to/unipora_teacher_checkpoint.pth
```

## Reuse Modes

The YAML flags support partial reruns:

- `PRUNING.COMPUTE_IMPORTANCE`
- `PRUNING.APPLY_PRUNING`
- `PRUNING.RUN_RECOVERY`
- `PRUNING.EVALUATE_AFTER_PRUNING`
- `PRUNING.IMPORTANCE.LOAD_PATH`
- `PRUNING.PRUNER.LOAD_MASK`
- `PRUNING.PRUNER.LOAD_PRUNED_CHECKPOINT`
- `PRUNING.RECOVERY.STUDENT_CHECKPOINT`

If you have an older stage 3 importance file that was generated before `token_delta_losses` was added, you must recompute stage 3 importance. The new loader will reject old stage 3 importance payloads instead of silently reusing them.

## Result Aggregation

Aggregate completed experiments with:

```bash
python scripts/summarize_pruning_results.py \
  --results-root output/promlora_tiny_448_r64_prom50_scale4_pertask/unipora_ta_prompt_pruning/pruning \
  --csv-out output/pruning_summary.csv \
  --md-out output/pruning_summary.md
```
