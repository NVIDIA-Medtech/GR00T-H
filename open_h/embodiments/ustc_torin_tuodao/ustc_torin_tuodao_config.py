"""
USTC Torin modality configuration for GR00T N1.6.

This configuration supports the USTC merged surgical dataset with:
- Stereo endoscope video (left/right)
- 14D joint-angle state (7 joints per arm)
- Cartesian absolute EEF pose from `action.cartesian_absolute` (xyz + quat + gripper per arm)
- REL_XYZ_ROT6D pose actions (xyz + rot6d) with absolute gripper

Important representation note:
The parquet files include `observation.state` (joint angles),
`observation.current_target_psm` (absolute pose per arm at t),
`action` (EEF deltas), and
`action.cartesian_absolute` (absolute pose per arm at t+1).
The modality mapping slices out pose-only values (7D per arm) and skips the
gripper slots in the cartesian absolute columns. REL_XYZ_ROT6D uses
`observation.current_target_psm` as the reference pose (t) and predicts pose
deltas toward `action.cartesian_absolute` (t+1). Gripper is modeled as an
ABSOLUTE action.

Data update note:
The raw conversion script writes `action.cartesian_absolute` alongside
`absolute_action` and repairs invalid rotations by carrying forward the last
valid rotation (identity if none), preserving translation.

Language:
Use per-frame `instruction.text` strings from the parquet files. This is mapped via
`annotation.instruction` in `open_h/embodiments/ustc_torin_tuodao/modality.json` with `is_text: true`, which
instructs the loader to pass through the raw text instead of using task indices.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Action horizon (how many future action steps to predict)
# 24 Hz -> 50 frames ~= 2.08 seconds
ACTION_HORIZON = 50

ustc_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope_left",
            # "endoscope_right", # Mono Only
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Single reference state for the current timestep
        modality_keys=[
            "left_joints",
            "right_joints",
            "left_pose",
            "right_pose",
        ],
        # Joint angles are continuous - mean/std normalization is appropriate
        mean_std_embedding_keys=[
            "left_joints",
            "right_joints",
        ],
        # Pose keys are consumed by REL_XYZ_ROT6D and should pass through
        pass_through_keys=[
            "left_pose",
            "right_pose",
        ],
    ),
    "action": ModalityConfig(
        # Start at +1 so index 0 is the state reference (CMR-style offset)
        delta_indices=list(range(0, ACTION_HORIZON)),
        modality_keys=[
            "left_pose",
            "left_gripper",
            "right_pose",
            "right_gripper",
        ],
        action_configs=[
            # Left pose: REL_XYZ_ROT6D from reference pose
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="left_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
            ),
            # Left gripper: absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
            # Right pose: REL_XYZ_ROT6D from reference pose
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
            ),
            # Right gripper: absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.instruction"],  # Uses instruction.text (raw strings)
    ),
}

# Register with USTC_TORIN_TUODAO tag for Torin surgical robot data
register_modality_config(ustc_config, embodiment_tag=EmbodimentTag.USTC_TORIN_TUODAO)
