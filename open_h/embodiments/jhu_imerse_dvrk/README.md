# JHU IMERSE dVRK

Surgical robot data collected from the da Vinci Research Kit (dVRK) at JHU's IMERSE lab.

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `jhu_imerse_dvrk` |
| **Config File** | `open_h/embodiments/jhu_imerse_dvrk/jhu_imerse_dvrk_config.py` |

## IMERSE Datasets (JHU)

- **NephFat**: Matches the standard dVRK 16D state/action layout; can reuse `open_h/embodiments/jhu_imerse_dvrk/jhu_imerse_dvrk_config.py` and `open_h/embodiments/jhu_imerse_dvrk/modality.json`. The dataset includes stereo endoscope and both wrists; the default modality maps `endoscope_left` plus both wrists.
- **star_IL**: Single-arm KUKA + endo360; use `open_h/embodiments/jhu_imerse_dvrk/jhu_imerse_star_il_config.py` with `open_h/embodiments/jhu_imerse_dvrk/modality_imerse_star_il.json`.

## Mono Configuration

The `jhu_imerse_dvrk_mono_config.py` provides a configuration variant for training
with a single endoscope view only (monocular), as opposed to the standard stereo
endoscope + wrist camera setup. It uses the `jhu_imerse_dvrk_mono` embodiment tag.
All other aspects (dual-arm REL_XYZ_ROT6D actions, 16D state/action format) remain
identical to the standard configuration. Use this when your dataset lacks wrist
camera views or when you want to evaluate endoscope-only performance.

## Data Format

### State (16D)

| Key | Dim | Description |
|-----|-----|-------------|
| `psm1_pose` | 7D | PSM1 xyz position (3D) + quaternion xyzw (4D) |
| `psm1_gripper` | 1D | PSM1 jaw angle |
| `psm2_pose` | 7D | PSM2 xyz position (3D) + quaternion xyzw (4D) |
| `psm2_gripper` | 1D | PSM2 jaw angle |

### Action (16D, horizon 50)

Actions use `REL_XYZ_ROT6D` for EEF poses and `ABSOLUTE` for grippers:

| Key | Rep | Description |
|-----|-----|-------------|
| `psm1_pose` | REL_XYZ_ROT6D | Relative translation + 6D rotation for PSM1 |
| `psm1_gripper` | ABSOLUTE | PSM1 jaw angle |
| `psm2_pose` | REL_XYZ_ROT6D | Relative translation + 6D rotation for PSM2 |
| `psm2_gripper` | ABSOLUTE | PSM2 jaw angle |

### Video (3 camera views)

| Modality Key | Raw Key |
|--------------|---------|
| `endoscope_left` | `observation.images.endoscope.left` |
| `wrist_left` | `observation.images.wrist.left` |
| `wrist_right` | `observation.images.wrist.right` |

### Language

| Modality Key | Source |
|--------------|--------|
| `annotation.human.task_description` | `task_index` |

## Dataset Preparation

Prepare datasets (copy modality JSON and compute normalization statistics) using the
centralized preparation script:

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag JHU_IMERSE_DVRK \
    --modality-json open_h/embodiments/jhu_imerse_dvrk/modality.json \
    /path/to/dataset1 /path/to/dataset2
```

This copies `modality.json` into each dataset's `meta/` directory and computes normalization statistics (`stats.json` and `temporal_stats.json`). Each dataset gets its own statistics file, but at training time these are merged across all datasets sharing the same embodiment tag — so each embodiment trains with a single unified set of normalization statistics.

## Open-Loop Evaluation

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path /path/to/your/dvrk-dataset \
    --embodiment-tag JHU_IMERSE_DVRK \
    --model-path /path/to/checkpoint \
    --traj-ids 0 1 2 \
    --action-horizon 50 \
    --steps 500
```

## Open-Loop Evaluation

Evaluate the finetuned model against ground truth trajectories:

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path /path/to/dvrk-dataset \
    --embodiment-tag JHU_IMERSE_DVRK \
    --model-path /path/to/checkpoint \
    --traj-ids 0 1 2 \
    --action-horizon 16 \
    --steps 500
```

## File Structure

```
open_h/embodiments/jhu_imerse_dvrk/
├── README.md                        # This file
├── jhu_imerse_dvrk_config.py        # Built-in config module (auto-registered)
├── jhu_imerse_dvrk_mono_config.py   # Mono (endoscope-only) built-in config module
├── jhu_imerse_star_il_config.py     # IMERSE star_IL built-in config module
├── modality.json                    # Data key mappings (copy to dataset/meta/)
└── modality_imerse_star_il.json     # IMERSE star_IL data key mappings
```
