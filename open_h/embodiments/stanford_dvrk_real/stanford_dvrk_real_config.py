"""
Stanford real-robot (dVRK) modality configuration for GR00T N1.6.

This configuration supports the Stanford dVRK real-robot datasets:
- Needle Transfer
- Tissue Retraction
- Peg Transfer

Data Format:
- State: 12D Cartesian EEF (2 arms × [xyz + roll/pitch/yaw]) + 2 grippers
- Action: 12D Cartesian absolute (same as state poses) + 2 grippers

Action Representation:
- EEF poses are converted to REL_XYZ_ROT6D using Euler RPY inputs (xyz + rpy)
- Grippers are ABSOLUTE jaw angles
- Euler convention: roll/pitch/yaw in radians with `xyz` extrinsic rotation

Video:
- Stereo endoscope views: camera_left, camera_right (540x960 @ 30Hz)
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


# Action horizon (30 FPS -> 50 steps = 1.67 seconds)
ACTION_HORIZON = 50

stanford_real_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope_left",
            # "endoscope_right", # Mono Only
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
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
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        action_configs=[
            # PSM1 pose: REL_XYZ_ROT6D with Euler RPY inputs
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm1_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            # PSM1 gripper: ABSOLUTE jaw angle
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
            # PSM2 pose: REL_XYZ_ROT6D with Euler RPY inputs
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            # PSM2 gripper: ABSOLUTE jaw angle
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
        modality_keys=["task"],
    ),
}

# Register with Stanford real-robot embodiment tag
register_modality_config(stanford_real_config, embodiment_tag=EmbodimentTag.STANFORD_DVRK_REAL)
