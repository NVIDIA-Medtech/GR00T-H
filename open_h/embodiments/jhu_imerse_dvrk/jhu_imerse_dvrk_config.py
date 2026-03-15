"""
dVRK (da Vinci Research Kit) modality configuration for GR00T N1.6.

This configuration supports dual-arm surgical robot (PSM1 and PSM2) with:
- REL_XYZ_ROT6D action representation for EEF poses
- Percentile-based normalization with per-dataset statistics
- 3 camera views (endoscope_left + wrist_left + wrist_right)

Data Format:
- State: 16D = psm1_pose(7: xyz + quat_xyzw) + psm1_jaw(1) + psm2_pose(7: xyz + quat_xyzw) + psm2_jaw(1)
- Action: 16D (same format as state, representing setpoints)

REL_XYZ_ROT6D Conversion:
- PSM pose actions are converted to REL_XYZ_ROT6D: translation and rotation relative to current EEF
- Output: 9D per arm (xyz_rel + rot6d_rel)
- Gripper actions remain absolute (jaw angle)
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
ACTION_HORIZON = 50

dvrk_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope_left",
            "wrist_left",
            "wrist_right",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Single reference state for REL_XYZ_ROT6D
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        mean_std_embedding_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),  # [0, 1, 2, ..., 50]
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        action_configs=[
            # PSM1 pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm1_pose",  # Reference state for relative conversion
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",  # Input actions are xyz + quaternion
                reference_rotation_format="quat",  # Reference state is also xyz + quaternion
            ),
            # PSM1 gripper: absolute jaw angle (no rotation conversion needed)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
            # PSM2 pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            # PSM2 gripper: absolute jaw angle (no rotation conversion needed)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Register with JHU_IMERSE_DVRK tag for surgical robot finetuning
register_modality_config(dvrk_config, embodiment_tag=EmbodimentTag.JHU_IMERSE_DVRK)
