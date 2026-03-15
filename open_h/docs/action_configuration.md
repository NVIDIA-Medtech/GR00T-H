# Action Configuration for GR00T-H

This document explains the REL_XYZ_ROT6D action representation, how to configure it for your embodiment, and how to handle the common case where absolute EEF poses live in the state column rather than the action column. For background on `ModalityConfig` basics (delta_indices, modality_keys, normalization keys), see [`getting_started/data_config.md`](../../getting_started/data_config.md).


## 1. Why REL_XYZ_ROT6D

Surgical robot datasets store actions in many formats. Some use quaternions (xyzw or wxyz), others use Euler angles or axis-angle vectors. Some record absolute EEF poses, others record delta commands or joint velocities. Training a single model across all of them requires a common representation.

GR00T-H unifies EEF actions into **REL_XYZ_ROT6D**:

- **Translation**: relative XYZ displacement from the current EEF position (3D)
- **Rotation**: relative rotation in 6D representation, specifically the first two columns of the rotation matrix (6D)
- **Total per end-effector key**: 9D (3D xyz_rel + 6D rot6d_rel)

Non-EEF actions (gripper open/close, energy buttons, jaw angles) stay **ABSOLUTE** and keep their original dimensionality. They aren't converted.

The conversion happens dynamically inside `StateActionProcessor`. You configure what your data provides, the processor handles the rest. The core math lives in `gr00t/data/state_action/pose.py`, specifically `convert_to_rel_xyz_rot6d()` for training and `convert_from_rel_xyz_rot6d()` for inference.


## 2. The Core Requirement: Absolute EEF Measurement

REL_XYZ_ROT6D computes the cumulative displacement from the current EEF pose (at the reference timestep) to each future timestep in the action chunk. You need an absolute EEF pose (xyz + rotation) accessible at each timestep.

This can come from:
- **Your action column directly**, if actions are absolute EEF setpoints (e.g., JHU IMERSE dVRK, Hamlyn dVRK)
- **Your state column**, if only state has the absolute EEF pose. This uses the "Copy EEF" pattern described in Section 5 (e.g., TUD TUNDRA UR5e, PolyU Sim)

The rotation component can be quaternion (4D), Euler angles (3D), or already rot6d (6D). Set `input_rotation_format` accordingly. If you have no absolute EEF poses anywhere in your data, you cannot use REL_XYZ_ROT6D. Consider joint-space RELATIVE actions instead.


## 3. ActionConfig Fields Reference

Every action modality key needs an `ActionConfig`. The fields are defined in `gr00t/data/types.py`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rep` | `ActionRepresentation` | (required) | `REL_XYZ_ROT6D` for EEF pose keys, `ABSOLUTE` for gripper/non-EEF |
| `type` | `ActionType` | (required) | `EEF` for end-effector, `NON_EEF` for everything else |
| `format` | `ActionFormat` | (required) | `XYZ_ROT6D` for EEF after conversion, `DEFAULT` for non-EEF |
| `state_key` | `str` or `None` | `None` | Which state key provides the EEF reference frame for relative conversion |
| `input_rotation_format` | `str` | `"quat"` | Rotation format in your action data: `"quat"`, `"rot6d"`, or `"euler"` |
| `input_quat_order` | `str` | `"xyzw"` | Quaternion ordering if input is quat: `"xyzw"` (scipy) or `"wxyz"` (scalar-first) |
| `reference_rotation_format` | `str` | `"rot6d"` | Rotation format in the reference state |
| `reference_quat_order` | `str` | `"xyzw"` | Quaternion ordering if reference is quat |
| `normalization_type` | `str` | `"temporal_meanstd"` | `"temporal_meanstd"` (recommended), `"meanstd"`, `"minmax"`, or `"skip"` |
| `hold_through_clutch` | `bool` | `False` | For ABSOLUTE actions only: hold last engaged value during clutch-out instead of zeroing. Used by CMR Versius; applicable to any embodiment with a controller clutching mechanism |
| `translation_scaling_key` | `str` or `None` | `None` | State key whose value scales relative translation. Used by CMR Versius where hand controller kinematics differ from instrument kinematics; applicable to any embodiment with such a scaling mismatch |
| `rotation_scaling_key` | `str` or `None` | `None` | State key whose value scales relative rotation angle. Same use case as `translation_scaling_key` |

A typical EEF config and a typical non-EEF config, side by side:

```python
# EEF pose (converted to REL_XYZ_ROT6D)
ActionConfig(
    rep=ActionRepresentation.REL_XYZ_ROT6D,
    type=ActionType.EEF,
    format=ActionFormat.XYZ_ROT6D,
    state_key="eef_pose",
    normalization_type="temporal_meanstd",
    input_rotation_format="quat",
    reference_rotation_format="quat",
    reference_quat_order="xyzw",
)

# Gripper (stays absolute)
ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
    normalization_type="meanstd",
)
```


## 4. Rotation Format Variations Across Open-H

Different embodiments store rotations differently. Here's what the actual configs use:

| Embodiment | `input_rotation_format` | `input_quat_order` | `reference_rotation_format` | `reference_quat_order` | Notes |
|---|---|---|---|---|---|
| TUD TUNDRA UR5e | `quat` | `xyzw` | `quat` | `xyzw` | Standard scipy convention |
| PolyU Sim | `quat` | `xyzw` | `quat` | `xyzw` | Standard scipy convention |
| JHU IMERSE dVRK | `quat` | `xyzw` | `quat` | `xyzw` | Standard scipy convention |
| Hamlyn dVRK (15Hz + 30Hz) | `quat` | `wxyz` | `quat` | `wxyz` | Scalar-first quaternion |
| Stanford dVRK Real | `euler` | (n/a) | `euler` | (n/a) | Euler RPY in radians, `xyz` extrinsic convention |
| CMR Versius | `quat` | `xyzw` | `quat` | `xyzw` | Quaternion with motion scaling keys |

If your dataset uses quaternions, check whether they're `xyzw` (scipy convention, most common in Open-H) or `wxyz` (scalar-first, used by Hamlyn and some other robotics frameworks). Getting this wrong causes silent corruption: rotations will be nonsensical, and the model will train on garbage without raising any errors.

If your dataset uses Euler angles, set `input_rotation_format="euler"`. The processor assumes `xyz` extrinsic convention (roll, pitch, yaw) in radians.


## 5. The Copy-EEF Pattern: Sourcing Actions from State

Some datasets store delta commands, joint velocities, or joint angles in the action column rather than absolute EEF poses. REL_XYZ_ROT6D needs absolute EEF poses at future timesteps. If the absolute EEF pose only exists in the state column, you can redirect the action loader to read from state instead.

### The Solution

Use the `original_key` field in `modality.json` to tell the data loader where to actually read from. Combined with `delta_indices=list(range(1, H+1))`, this reads future states as action targets.

### Example 1: TUD TUNDRA UR5e

The TUD dataset's action column contains delta commands (`dx, dy, dz, droll`), which can't be used for REL_XYZ_ROT6D directly. The absolute EEF pose lives in `observation.state` at indices 26-33.

The modality JSON (`modality_grasping_retraction.json`):

```json
{
    "state": {
        "joint_position": {"start": 8, "end": 14, "original_key": "observation.state"},
        "eef_pose": {"start": 26, "end": 33, "original_key": "observation.state"}
    },
    "action": {
        "eef_pose": {"start": 26, "end": 33, "original_key": "observation.state"},
        "gripper": {"start": 4, "end": 5, "original_key": "action"}
    }
}
```

How it works:

- `action.eef_pose` has `"original_key": "observation.state"`. This tells the data loader to read from the state column, not the action column.
- The indices `[26:33]` select the 7D EEF pose (xyz + quaternion) from the 33D state vector.
- The config sets `delta_indices=list(range(1, 51))`. This reads `state[t+1], state[t+2], ..., state[t+50]`, creating a 50-step trajectory of absolute EEF poses from future timesteps.
- `action.gripper` uses `"original_key": "action"` to read from the actual action column. Ensure the gripper values at each timestep are aligned with the corresponding action timestep (e.g., `gripper[t]` corresponds to `state[t+1]` when using `delta_indices` starting from 1).
- The `+1` offset in delta_indices (starting from 1, not 0) means the first action target is the next timestep's pose.

### Example 2: PolyU Sim

The PolyU simulation stores joint deltas in its action column, but absolute Cartesian pose lives in a separate `observation.cartesian_state` column.

The modality JSON:

```json
{
    "state": {
        "psm_joints": {"start": 0, "end": 10},
        "psm_cartesian_pose": {"start": 0, "end": 7, "original_key": "observation.cartesian_state"}
    },
    "action": {
        "psm_cartesian_pose": {"start": 0, "end": 7, "original_key": "observation.cartesian_state"},
        "psm_gripper": {"start": 10, "end": 11, "original_key": "action"}
    }
}
```

Same pattern, but reading from a different source column (`observation.cartesian_state` instead of `observation.state`). The `original_key` can point to any column in your parquet file. This flexibility lets you source action data from whichever column actually contains the absolute EEF pose.


## 6. The state_key and pass_through_keys

`ActionConfig.state_key = "eef_pose"` tells the processor: use this state key as the reference frame for relative conversion. The reference pose at the current timestep becomes the origin, and all future EEF poses in the action chunk are expressed as cumulative displacements from it.

`pass_through_keys` is an optional field on `ModalityConfig` that lists state keys which are loaded from the dataset for intermediate calculations (such as providing the REL_XYZ_ROT6D reference frame) but are **not** sent to the model's state encoder. Example:

```python
"state": ModalityConfig(
    delta_indices=[0],
    modality_keys=["joint_position", "eef_pose"],
    mean_std_embedding_keys=["joint_position"],
    pass_through_keys=["eef_pose"],  # loaded for processing, NOT sent to model
)
```

In this example, `joint_position` is embedded by the state encoder and fed to the model. `eef_pose` is used only for the REL_XYZ_ROT6D conversion and then discarded. The `state_key` referenced by `ActionConfig` does not have to be in `pass_through_keys` — if it isn't, it will be both embedded as model input and used as the conversion reference.

Note that GR00T-H was trained with `state_dropout_prob_per_embodiment: 1.0` for all embodiments in `gr00t_h_config.yaml`. This zeros out all state inputs before the encoder, making the model vision-only at inference. The `pass_through_keys` mechanism is separate from state dropout — it controls which keys reach the encoder, while state dropout controls whether the encoder output is used.


## 7. Dual-Arm Configurations

For multi-arm robots, each arm gets its own modality key and `ActionConfig`. Here's the structure from JHU IMERSE dVRK (dual-arm da Vinci):

```python
"action": ModalityConfig(
    delta_indices=list(range(50)),
    modality_keys=[
        "psm1_pose",
        "psm1_gripper",
        "psm2_pose",
        "psm2_gripper",
    ],
    action_configs=[
        ActionConfig(rep=ActionRepresentation.REL_XYZ_ROT6D, type=ActionType.EEF,
                     format=ActionFormat.XYZ_ROT6D, state_key="psm1_pose",
                     input_rotation_format="quat", reference_rotation_format="quat",
                     normalization_type="temporal_meanstd"),
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                     format=ActionFormat.DEFAULT, normalization_type="temporal_meanstd"),
        ActionConfig(rep=ActionRepresentation.REL_XYZ_ROT6D, type=ActionType.EEF,
                     format=ActionFormat.XYZ_ROT6D, state_key="psm2_pose",
                     input_rotation_format="quat", reference_rotation_format="quat",
                     normalization_type="temporal_meanstd"),
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                     format=ActionFormat.DEFAULT, normalization_type="temporal_meanstd"),
    ],
)
```

Key rules:
- `action_configs` must have the same length as `modality_keys`. One `ActionConfig` per key, in order.
- Each EEF `ActionConfig` points to its own `state_key` for the reference frame (`"psm1_pose"` for PSM1, `"psm2_pose"` for PSM2).
- Non-EEF keys (grippers) don't need a `state_key`.

The SanoScience Sim embodiment extends this to 4 arms (8 action configs). The pattern scales naturally.


## 8. Adding Your Own Embodiment: Decision Guide

### Step 1: Do you have absolute EEF poses?

- **In your action column** (e.g., absolute Cartesian setpoints): map the action keys directly in modality.json using `start`/`end` indices. No `original_key` is needed — the loader reads from the `action` column by default. See `jhu_imerse_dvrk/modality.json` for an example.
- **Only in your state column**: use the Copy-EEF pattern from Section 5. Set `"original_key": "observation.state"` (or whatever column contains the EEF pose). Then, determine the proper `delta_indices` offset to use, such that `action[t] = state[t+1]` 
- **Nowhere** (only joint angles or delta commands with no FK available): you cannot use REL_XYZ_ROT6D for EEF actions. Consider joint-space RELATIVE representation instead.

### Step 2: What rotation format does your data use?

- Quaternion xyzw: `input_rotation_format="quat"`, `input_quat_order="xyzw"`
- Quaternion wxyz: `input_rotation_format="quat"`, `input_quat_order="wxyz"`
- Already 6D rotation: `input_rotation_format="rot6d"`
- Euler angles (radians, xyz extrinsic): `input_rotation_format="euler"`
- Euler angles in some other convention: convert to one of the above during dataset creation

### Step 3: Single arm or multi-arm?

- **Single arm**: one EEF key + one gripper key (if applicable). See TUD TUNDRA UR5e or PolyU Sim.
- **Dual arm**: separate keys per arm. See JHU IMERSE dVRK or Hamlyn dVRK.
- **More than two**: same pattern, more keys. See SanoScience Sim (4 instruments) or Rob Surgical BiTrack (3 arms).

### Step 4: Create your config files

1. **Add your embodiment tag** to `gr00t/data/embodiment_tags.py` (add a new entry to the `EmbodimentTag` enum).
2. **Add a projector index** in `gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py` (maps your tag to a unique projector slot).
3. **Create your config directory**: `open_h/embodiments/your_robot/`
4. **Write `your_robot_config.py`**: define the `ModalityConfig` dict with video, state, action, and language entries. Call `register_modality_config()` at module level.
5. **Write `modality.json`**: map your parquet columns and indices to named keys. Use `original_key` to redirect reads when needed.
6. **Run `prepare_datasets.sh`** to generate `stats.json` and `temporal_stats.json`.

### Reference configs by pattern

| Pattern | Example embodiment | Key features |
|---------|-------------------|--------------|
| Single arm, copy-EEF | `tud_tundra_ur5e` | Actions sourced from state column, quaternion xyzw |
| Dual arm, actions in action column | `jhu_imerse_dvrk` | Direct action column, quaternion xyzw |
| Dual arm, wxyz quaternions | `hamlyn_dvrk` | Scalar-first quaternion ordering |
| Dual arm, Euler angles | `stanford_dvrk_real` | Euler RPY input, xyz extrinsic |
| Dual arm, clutch-aware | `cmr_versius` | Motion scaling, hold-through-clutch, engagement filtering |
| Single arm, simulated | `polyu_sim` | Copy-EEF from separate cartesian_state column |
