# Obuda dVRK

Obuda University dVRK surgical datasets for Open-H tasks:
- FRS Dome (knot tying, suturing)
- Needle Threading
- Peg Transfer
- Rollercoaster
- Seaspike

All raw datasets are LeRobot v2.1 with 30Hz video and 23D state/action.

---

## Data Summary

### Datasets

| Dataset | Episodes | Tasks |
| --- | --- | --- |
| `FRS_Dome_1` | 102 | Knot tying, Suturing |
| `NeedleThreading_1` | 196 | Needle Threading |
| `PegTransfer_1` | 216 | Peg Transfer |
| `Rollercoaster_1` | 95 | Rollercoaster |
| `Seaspike_1` | 207 | Seaspike |


### Cameras

| View | Original Key |
| --- | --- |
| `endoscope_left` | `observation.images.endoscope.left` |
| `wrist_left` | `observation.images.wrist.left` |
| `wrist_right` | `observation.images.wrist.right` |

### State (16D)

We use PSM1/PSM2 EEF poses only (ECM excluded):
- `psm1_pose` (7D): xyz + quaternion (xyzw)
- `psm1_gripper` (1D): jaw angle
- `psm2_pose` (7D): xyz + quaternion (xyzw)
- `psm2_gripper` (1D): jaw angle

### Action (16D)

Same layout as state, with absolute setpoints:
- EEF poses converted to `REL_XYZ_ROT6D` (xyz_rel + rot6d_rel)
- Grippers remain absolute

---

## Modality Mapping

Use `open_h/embodiments/obuda_dvrk/modality.json`:
- State: `psm1_pose`, `psm1_gripper`, `psm2_pose`, `psm2_gripper`
- Action: `psm1_pose`, `psm1_gripper`, `psm2_pose`, `psm2_gripper`
- Video: `endoscope_left`, `wrist_left`, `wrist_right`
- Language: `task` (from `task_index`)

---

## Embodiment Configuration

Embodiment tag: `OBUDA_DVRK`

Action configuration:
- `REL_XYZ_ROT6D` for EEF poses (quaternion -> rot6d)
- `ABSOLUTE` for grippers
- `ACTION_HORIZON = 50` (30Hz -> 1.67s)

See `open_h/embodiments/obuda_dvrk/obuda_dvrk_config.py`.

---

## Dataset Preparation

Generate stats files (copies modality JSON into `meta/`, generates `stats.json` and `temporal_stats.json`):

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag OBUDA_DVRK \
    --modality-json open_h/embodiments/obuda_dvrk/modality.json \
    /path/to/obuda_dataset
```

Repeat for each sub-dataset (FRS_Dome_1, NeedleThreading_1, etc.).

---

---

## File Structure

```
open_h/embodiments/obuda_dvrk/
  modality.json            # State/action/video field mapping
  obuda_dvrk_config.py     # Embodiment config (action configs, modality keys)
  README.md                # This file
```
