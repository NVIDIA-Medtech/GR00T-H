# Turin MITIC Ex Vivo Embodiment

Dual-arm dVRK (PSM1/PSM2) ex vivo surgical demonstrations covering knot tying, needle manipulation, and tissue lift tasks, recorded with stereo endoscope views.

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `TURIN_MITIC_EX_VIVO` |
| **Config File** | `open_h/embodiments/turin_mitic_ex_vivo/turin_mitic_ex_vivo_config.py` |
| **Modality Mapping** | `open_h/embodiments/turin_mitic_ex_vivo/modality.json` |

## Data Summary

| Property | Value |
|----------|-------|
| Source | Turin (MITIC) ex vivo surgical demonstrations |
| Robot | dVRK (dual-arm PSM1/PSM2) |
| Tasks | Knot tying, needle manipulation, tissue lift |
| Episodes | 799 |
| Frames | 388,690 |
| FPS | 30 |
| Video | 1080x1920, AV1 |

## Data Format

### State

- `psm1_joints` (6D) -- PSM1 joint angles from `observation.state[0:6]`, embedded via mean/std
- `psm2_joints` (6D) -- PSM2 joint angles from `observation.state[6:12]`, embedded via mean/std
- `psm1_pose` (7D) -- PSM1 EEF pose (xyz + quat) from `action[0:7]`, pass-through reference for REL_XYZ_ROT6D (not embedded)
- `psm2_pose` (7D) -- PSM2 EEF pose (xyz + quat) from `action[7:14]`, pass-through reference for REL_XYZ_ROT6D (not embedded)

### Action

- `psm1_pose` (7D) -- PSM1 absolute EEF pose (xyz + quaternion)
- `psm2_pose` (7D) -- PSM2 absolute EEF pose (xyz + quaternion)

Action representation: `REL_XYZ_ROT6D` for both arms (quaternion input -> relative xyz + rot6d output = 9D per arm per timestep).
Action horizon: 50 (~1.67s at 30 Hz). `delta_indices = 1..H`.

### Cameras

- `endoscope_left`
- Note: `endoscope_right` is available in the dataset but only the left view is used in the config (mono only).

### Language

- `annotation.instruction` (mapped from `instruction.text`)

## Action Horizon

- 50 steps (30 Hz, ~1.67 s of future prediction)

## Dataset Preparation

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag TURIN_MITIC_EX_VIVO \
    --modality-json open_h/embodiments/turin_mitic_ex_vivo/modality.json \
    /path/to/turin_dataset
```

## Notes
- No EEF pose is present in `observation.state`. The action at `t=0` is used as the REL_XYZ_ROT6D reference pose via pass-through keys.
- Dataset synchronization uses the right endoscope timestamp; `action` comes from measured Cartesian pose topics aligned to that frame, so `action[t]` is expected to be t-aligned (not t+1) with `observation.state[t]` and images.
- Only endoscope views are available; no wrist cameras.
