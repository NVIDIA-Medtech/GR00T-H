# PolyU Simulated Surgical Embodiment

Tissue retraction task in laparoscopic cholecystectomy using the PolyU Open-H surgical simulation dataset (msr_surgical robot).

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `POLYU_SIM` |
| **Config File** | `open_h/embodiments/polyu_sim/polyu_sim_config.py` |
| **Modality Mapping** | `open_h/embodiments/polyu_sim/modality.json` |

## Data Summary

| Property | Value |
|----------|-------|
| Source | PolyU (Open-H Surgical) |
| Task | Tissue Retraction in Laparoscopic Cholecystectomy |
| Episodes | 11,520 |
| Frames | 5,760,000 |
| FPS | 30 |

## Data Format

### State
- 10D model input + 7D pass-through:
  - `psm_joints` (10D): joint angles from `observation.state`
  - `psm_cartesian_pose` (7D, **pass-through**): xyz + quat_xyzw from `observation.cartesian_state` -- used as the reference state for REL_XYZ_ROT6D action conversion but not fed to the model as state input

### Action
- 8D total:
  - `psm_cartesian_pose` (7D): pose from `observation.cartesian_state` (aligned to t+1)
  - `psm_gripper` (1D): gripper signal from `action` (likely all values are 1.0)

### Cameras
- `endoscope`: `observation.images.main` (1280x720)

### Language
- `task` from `meta/tasks.jsonl`

## Action Horizon
- 50 steps @ 30 Hz (~1.7 s look-ahead)

## Dataset Preparation

Use the shared preparation script to copy the modality JSON into each dataset's `meta/` folder and generate normalization statistics:

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag polyu_sim \
    --modality-json open_h/embodiments/polyu_sim/modality.json \
    /path/to/dataset
```

## Notes
- Pose actions are converted to REL_XYZ_ROT6D using `psm_cartesian_pose` as the reference state.
- Joint deltas in `action[:10]` are present in the raw dataset, but the config uses pose actions.
