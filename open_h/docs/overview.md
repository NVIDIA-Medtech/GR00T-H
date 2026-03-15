# GR00T-H Overview

## Intended Use

GR00T-H is intended for use in robotics R&D, including exploration of surgical robotics and robotic ultrasound policies, benchmarking, and method development. It is not intended for clinical deployment, patient care, or medical decision-making.

GR00T-H is not expected to work out of the box for arbitrary embodiments or tasks. It may produce reasonable behavior for the specific robots and tasks represented in the Open-H dataset, but the primary value of this release is as a pretrained VLA checkpoint for healthcare robotics. The expected workflow is to **finetune from GR00T-H** on your own robot's data — not to deploy it zero-shot. Zero-shot deployment on new embodiments or tasks is unlikely to produce usable results.

## What Changed from Core GR00T

This repository is a superset of upstream [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T). The table below summarizes every file added or modified.

| File | Change |
|------|--------|
| `open_h/` (new package) | All embodiment configs, training YAML, dataset prep script |
| `gr00t/data/embodiment_tags.py` | 20 new surgical robot tags added to `EmbodimentTag` enum |
| `gr00t/data/types.py` | `REL_XYZ_ROT6D` action representation, clutch/motion-scaling fields, temporal normalization type |
| `gr00t/data/state_action/pose.py` | `convert_to_rel_xyz_rot6d()` / `convert_from_rel_xyz_rot6d()` conversion math |
| `gr00t/data/state_action/state_action_processor.py` | REL_XYZ_ROT6D conversion path, clutch-aware zeroing, motion scaling, temporal normalization |
| `gr00t/data/stats.py` | Temporal stats generation (per-timestep normalization for action chunks) |
| `gr00t/experiment/launch_finetune.py` | Auto-registers Open-H configs on import, adds `--calculate-norm-stats` mode |
| `gr00t/experiment/launch_train.py` | Auto-registers Open-H configs on import, refactored config loading |
| `gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py` | Projector index mappings for all Open-H embodiments |

## Embodiment Auto-Registration

All Open-H embodiment configs are auto-registered at import time. When any training or stats script runs, `open_h/embodiments/__init__.py` discovers and executes every `*_config.py` file, which calls `register_modality_config()` for each embodiment tag. This means:

- **Built-in Open-H embodiment**: pass `--embodiment-tag <TAG>` only. Do not pass `--modality-config-path` — the config is already registered and doing so will raise an `AssertionError`.
- **Brand-new robot not in Open-H**: pass `--embodiment-tag NEW_EMBODIMENT --modality-config-path your_config.py`.

Note: the `--modality-json` flag in `prepare_datasets.sh` is unrelated — it points to the JSON column mapping, not the Python config.

## Quick Start

### A. Finetune a built-in Open-H embodiment (single dataset)

Four steps: prepare dataset statistics, verify data with replay, finetune, evaluate.

```bash
# 1. Prepare dataset (copies modality.json, generates stats.json + temporal_stats.json)
bash open_h/prepare_datasets.sh \
    --embodiment-tag TUD_TUNDRA_UR5E \
    --modality-json open_h/embodiments/tud_tundra_ur5e/modality_grasping_retraction.json \
    /path/to/tud_dataset
```

```bash
# 2. Replay recorded actions to verify data processing and action conversion
uv run python gr00t/eval/run_gr00t_server.py \
    --dataset-path /path/to/tud_dataset \
    --embodiment-tag TUD_TUNDRA_UR5E \
    --execution-horizon 8
```

Run this before any training to confirm your embodiment config loads correctly and actions are converted as expected. The server replays ground-truth actions from the dataset through the full processing pipeline (modality loading, REL_XYZ_ROT6D conversion, normalization). See the [Policy Guide](../../getting_started/policy.md#debugging-with-replaypolicy) for client-side usage and episode switching.

```bash
# 3. Finetune from GR00T-H checkpoint
uv run torchrun --nproc_per_node=8 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-H \
    --dataset-path /path/to/tud_dataset \
    --embodiment-tag TUD_TUNDRA_UR5E \
    --num-gpus 8 \
    --global-batch-size 32 \
    --max-steps 20000 \
    --output-dir /path/to/output
```

```bash
# 4. Open-loop evaluation
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path /path/to/tud_dataset \
    --embodiment-tag TUD_TUNDRA_UR5E \
    --model-path /path/to/output/checkpoint-20000 \
    --traj-ids 0 \
    --action-horizon 50 \
    --steps 400
```

Note: no `--modality-config-path` anywhere. The config is already registered.

The evaluation script writes trajectory visualizations to `/tmp/open_loop_eval/` showing predicted vs. ground-truth actions with per-dimension MSE.

### B. Multi-embodiment training

The released GR00T-H checkpoint was trained on 4 nodes with 8 GPUs each, using the config at `open_h/gr00t_h_config.yaml`.

Before launching, replace every `REPLACE_WITH_OPEN_H_DATA_PATH` in the YAML with your local data root.

```bash
uv run torchrun --nnodes=4 --nproc_per_node=8 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    gr00t/experiment/launch_train.py \
    --load-config-path open_h/gr00t_h_config.yaml
```

For fewer GPUs, adjust both `global_batch_size` and `num_gpus` in the YAML accordingly. On a single 8-GPU node, set `num_gpus: 8` and `global_batch_size: 256`.

### C. Add your own healthcare embodiment

If your robot isn't already in Open-H, you'll need to:

1. Convert your data to LeRobot v2 format
2. Write a `modality.json` mapping your dataset columns to named keys
3. Define a `ModalityConfig` with `ActionConfig` entries for your end-effector(s)
4. Run the stats pipeline to generate normalization statistics

See [Action Configuration](action_configuration.md) for how to write a config and [Data Preparation](data_preparation.md) for the stats pipeline.

## Further Reading

| Topic | Document |
|-------|----------|
| Action config deep-dive | [Action Configuration](action_configuration.md) |
| Data preparation pipeline | [Data Preparation](data_preparation.md) |
| Embodiment comparison table | [Embodiment Overview](../embodiments/README.md) |
| LeRobot v2 format | [Data Preparation Guide](../../getting_started/data_preparation.md) |
| Base ModalityConfig reference | [Data Config Guide](../../getting_started/data_config.md) |
| Inference / Policy API | [Policy Guide](../../getting_started/policy.md) |
| Core GR00T finetuning | [Finetune New Embodiment](../../getting_started/finetune_new_embodiment.md) |
