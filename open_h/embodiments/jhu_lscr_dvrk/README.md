# JHU LSCR (Open-H Surgical)

JHU LSCR is a collection of dVRK-based surgical datasets spanning three sub-benchmarks:
ARCADE (cartesian EEF setpoints), MIRACLE (cartesian EEF actions), and SMARTS
(cartesian EEF actions with additional cameras).

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tags** | `JHU_IMERSE_DVRK` (ARCADE), `JHU_LSCR_DVRK_MIRACLE`, `JHU_LSCR_DVRK_SMARTS` |
| **Config File** | `open_h/embodiments/jhu_lscr_dvrk/jhu_lscr_dvrk_config.py` |

ARCADE uses the standard dVRK embodiment config; MIRACLE and SMARTS are registered
in the LSCR config file above.

## Data Format

### State

| Sub-benchmark | Keys | Notes |
|---------------|------|-------|
| ARCADE | `observation.state` / `action` | EEF pose + gripper via dVRK schema |
| MIRACLE | `psm1_pose` (7), `psm1_gripper` (1), `psm2_pose` (7), `psm2_gripper` (1) | PSM cartesian pose per arm |
| SMARTS | `psm1_pose` (7), `psm1_gripper` (1), `psm2_pose` (7), `psm2_gripper` (1) | Per-key arrays (not a single `observation.state` vector) |

### Action

All sub-benchmarks use `REL_XYZ_ROT6D` for EEF pose actions and `ABSOLUTE` for
non-EEF (gripper). MIRACLE and SMARTS use `delta_indices` starting at `t+1`
(reference pose at `t`, actions at `t+1..t+H`), while ARCADE starts at `t+0`
(using the standard dVRK config).

| Sub-benchmark | Action Horizon | FPS |
|---------------|----------------|-----|
| ARCADE | 50 | 30 Hz |
| MIRACLE | 25 | 15 Hz |
| SMARTS | 16 | 10 Hz |

### Cameras

| Sub-benchmark | Camera Streams (used in training) |
|---------------|----------------------------------|
| ARCADE | `endoscope_left`, `wrist_left`, `wrist_right` |
| MIRACLE | `camera_left` (mono endoscope) |
| SMARTS | `endoscope_left`, `camera_side_view` |

Note: MIRACLE also has `camera_right` and SMARTS has `endoscope_right` in their
modality JSONs, but these are commented out in the config (mono only).

### Language

All configs use `tasks.jsonl` via `task_index`. ARCADE uses
`annotation.human.task_description`; MIRACLE/SMARTS use `annotation.task`.

## Modality Mapping

Each sub-benchmark has its own modality JSON:

- **ARCADE**: `modality_arcade.json` -- EEF pose + gripper from `observation.state`/`action`, annotation key aligned to the dVRK schema.
- **MIRACLE**: `modality_miracle.json` -- PSM cartesian pose + grippers from `observation.state` slices.
- **SMARTS**: `modality_smarts.json` -- PSM cartesian pose + grippers from per-key arrays.

## Dataset Preparation

Run `prepare_datasets.sh` for each sub-benchmark with the matching modality file
(the script copies the modality JSON into `meta/` automatically):

```bash
# ARCADE (30 Hz, action horizon 50)
bash open_h/prepare_datasets.sh \
    --embodiment-tag JHU_IMERSE_DVRK \
    --modality-json open_h/embodiments/jhu_lscr_dvrk/modality_arcade.json \
    /path/to/JHU_LSCR/ARCADE

# MIRACLE (15 Hz, action horizon 25)
bash open_h/prepare_datasets.sh \
    --embodiment-tag JHU_LSCR_DVRK_MIRACLE \
    --modality-json open_h/embodiments/jhu_lscr_dvrk/modality_miracle.json \
    /path/to/JHU_LSCR/MIRACLE

# SMARTS (10 Hz, action horizon 16)
bash open_h/prepare_datasets.sh \
    --embodiment-tag JHU_LSCR_DVRK_SMARTS \
    --modality-json open_h/embodiments/jhu_lscr_dvrk/modality_smarts.json \
    /path/to/JHU_LSCR/SMARTS
```

## Notes

- MIRACLE publishes full PSM cartesian pose (XYZ + quaternion) per arm, so the modality mapping uses EEF pose keys and REL_XYZ_ROT6D actions to avoid losing pose information.
- ARCADE includes `instruction.text` in the dataset, but tasks are currently sourced from `tasks.jsonl` for consistency across LSCR datasets.
- SMARTS uses separate per-key arrays for cartesian pose and grippers, not a single `observation.state` vector.
- ARCADE/cautery is missing videos for episodes 12-21 across all camera streams; training should exclude a `missing_videos` split once it is defined in `meta/info.json`.
