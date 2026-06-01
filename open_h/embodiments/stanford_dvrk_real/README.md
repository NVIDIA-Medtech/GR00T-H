# Stanford Real Robot (dVRK)

Stanford real-robot dVRK datasets for Open-H surgical tasks:
- Needle Transfer
- Tissue Retraction
- Peg Transfer

All datasets are LeRobot v2.1 with 30Hz stereo endoscopic video.

---

## Data Summary

### Episodes and Frames (from `meta/info.json`)
- Needle Transfer: 700 episodes, 313,882 frames
- Tissue Retraction: 698 episodes, 291,826 frames
- Peg Transfer: 598 episodes, 268,729 frames

### Cameras
- `observation.images.camera_left` (540x960, 30Hz)
- `observation.images.camera_right` (540x960, 30Hz)

### State (`observation.state`, shape=26)
We only use the EEF pose + gripper for each arm:
- PSM1 gripper: index 6
- PSM1 EEF pose (xyz + roll/pitch/yaw): indices 7-12
- PSM2 gripper: index 19
- PSM2 EEF pose (xyz + roll/pitch/yaw): indices 20-25

### Action (`action`, shape=14)
Absolute Cartesian EEF pose + gripper in camera/ECM frame:
- PSM1 gripper: index 0
- PSM1 EEF pose (xyz + roll/pitch/yaw): indices 1-6
- PSM2 gripper: index 7
- PSM2 EEF pose (xyz + roll/pitch/yaw): indices 8-13

Orientation uses Euler RPY (roll, pitch, yaw) in radians. We assume `xyz` extrinsic.

---

## Modality Mapping

Use `open_h/embodiments/stanford_dvrk_real/modality_real_robot.json`:
- State: `psm1_pose`, `psm1_gripper`, `psm2_pose`, `psm2_gripper`
- Action: `psm1_pose`, `psm1_gripper`, `psm2_pose`, `psm2_gripper`
- Video: `endoscope_left` (mono; `endoscope_right` is defined in `modality_real_robot.json` but disabled in the current config)
- Language: `task` (from `task_index`)

---

## Embodiment Configuration

Embodiment tag: `stanford_dvrk_real`

Action configuration:
- `REL_XYZ_ROT6D` for EEF pose (Euler RPY -> rot6d)
- `ABSOLUTE` for grippers
- `ACTION_HORIZON = 50` (30Hz -> 1.67s)

See `open_h/embodiments/stanford_dvrk_real/stanford_dvrk_real_config.py`.

---

## Split Policy (Keep Recovery, Exclude Fail)

Needle Transfer and Peg Transfer include `recovery` and `fail` splits in `info.json`.
We **keep recovery** but **exclude fail** using runtime split filtering:

In training configs (YAML):
```yaml
exclude_splits: ["fail"]
```

This applies to both training and stats generation without copying datasets.

---

## Stats Generation

Generate stats files (copies modality, generates `stats.json` and `temporal_stats.json`):

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag STANFORD_DVRK_REAL \
    --modality-json open_h/embodiments/stanford_dvrk_real/modality_real_robot.json \
    /path/to/stanford_dataset
```

Repeat for each task subset (Needle Transfer, Tissue Retraction, Peg Transfer).

---

## Notes
- `next.done` is **not** present in Stanford real-robot datasets, so no terminal padding filter is needed.
- If joint states become useful later, extend modality config to include joint groups from `observation.state`.
