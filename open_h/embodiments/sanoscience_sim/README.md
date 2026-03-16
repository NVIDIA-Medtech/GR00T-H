# SanoScience Sim

Simulated surgical robot data from SanoScience with 4-instrument control.

## Dataset Statistics

> Dataset: SanoScience v1.2 merged (480p, cleaned) — 5 training groups

| Property | Value |
|----------|-------|
| **Episodes** | 14,004 |
| **Total Frames** | 1,574,892 |
| **Avg Frames/Episode** | ~112 |

Training groups:
- Expert_demonstrations: 5,454 episodes, 603,054 frames (25 fps)
- NonExpert_full_modalities_clean_final: 6,156 episodes, 713,070 frames (30 fps)
- NonExpert_partial_modalities_clean_final: 666 episodes, 58,752 frames (30 fps)
- NonExpert_recovery_clean_final: 126 episodes, 20,376 frames (30 fps)
- NonExpert_stereo_clean_final: 1,602 episodes, 179,640 frames (30 fps)

### Episode Length Distribution

**Key stats:** Min=16, Max=669, Median=102, Mean=112.5, Q25=90, Q75=122

### Action Horizon Selection

Episodes shorter than the action horizon are **skipped during training** (not padded). We use `ACTION_HORIZON = 36`:

- 13,968 / 14,004 episodes usable (99.7%) — only 36 episodes dropped (< 36 frames)
- Usable training steps: 1,085,094

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `sanoscience_sim` |
| **Config File** | `open_h/embodiments/sanoscience_sim/sanoscience_sim_config.py` |



## Data Format

### State (32D)

The state vector is organized by instrument, with each instrument having a 7D pose and 1D gripper:
State is sourced from `action.cartesian_absolute` via `modality.json` to keep state/action formats consistent.

| Instrument | Indices | Dimensions | Description |
|------------|---------|------------|-------------|
| inst_0 | 0-7 | 7D + 1D | xyz (3D) + quaternion xyzw (4D) + gripper_angle_rad (1D) |
| inst_1 | 8-15 | 7D + 1D | xyz (3D) + quaternion xyzw (4D) + gripper_angle_rad (1D) |
| inst_2 | 16-23 | 7D + 1D | xyz (3D) + quaternion xyzw (4D) + gripper_angle_rad (1D) |
| inst_3 | 24-31 | 7D + 1D | xyz (3D) + quaternion xyzw (4D) + gripper_angle_rad (1D) |

### Action (32D)

Same format as state. Actions are extracted from the `action.cartesian_absolute` column (specified in `modality.json`).

After REL_XYZ_ROT6D conversion, the action output becomes 40D:
- 4 instruments x (9D REL_XYZ_ROT6D pose + 1D gripper)
- Pose: xyz_rel (3D) + rot6d_rel (6D) = 9D
- Gripper: absolute angle (1D)

Action horizon and indexing:
- `ACTION_HORIZON = 36` — 99.7% of episodes usable.
- `delta_indices = [1, 2, 3, ..., 36]` so index 0 is the state reference (CMR-style offset).

### Video (1 view)

| Camera | Original Key |
|--------|--------------|
| `camera_color` | `observation.images.color` |

## Dataset Preparation

Generate normalization statistics (from repo root):

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag SANOSCIENCE_SIM \
    --modality-json open_h/embodiments/sanoscience_sim/modality.json \
    /path/to/sanoscience_dataset
```

## Training

### Single-Dataset Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /path/to/your/sanoscience-dataset \
    --embodiment-tag SANOSCIENCE_SIM \
    --num-gpus 1 \
    --output-dir /path/to/output/checkpoints \
    --save-steps 2000 \
    --save-total-limit 5 \
    --max-steps 50000 \
    --warmup-ratio 0.05 \
    --weight-decay 1e-5 \
    --learning-rate 1e-4 \
    --use-wandb \
    --global-batch-size 32
```

## Open-Loop Evaluation

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path /path/to/sanoscience-dataset \
    --embodiment-tag SANOSCIENCE_SIM \
    --model-path /path/to/checkpoint \
    --traj-ids 0 1 2 \
    --action-horizon 36 \
    --steps 500
```

## File Structure

```
open_h/embodiments/sanoscience_sim/
├── README.md              # This file
├── sanoscience_sim_config.py  # Built-in config module (auto-registered)
└── modality.json          # Data key mappings (copy to dataset/meta/)
```
