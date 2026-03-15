"""
Obuda dVRK modality configuration for GR00T N1.6.

This configuration targets the Obuda University Open-H surgical datasets:
- FRS_Dome_1 (knot tying, suturing)
- NeedleThreading_1
- PegTransfer_1
- Rollercoaster_1
- Seaspike_1

Design decisions:
- State/action use absolute Cartesian EEF poses for PSM1/PSM2 only (16D).
- ECM pose is excluded in v1 to align with standard dVRK state/action size.
- Cameras use endoscope.left + both wrist views (3 views).
- Actions use REL_XYZ_ROT6D for EEF poses and ABSOLUTE for grippers.
- Language uses tasks.jsonl via the "task" key.
- ACTION_HORIZON = 50 (all episodes are >= 50 frames).
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


# Action horizon (how many future steps to predict).
# We keep the 50-step horizon to match other Open-H surgical configs.
ACTION_HORIZON = 50

obuda_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope_left",
            "wrist_left",
            "wrist_right",
        ],
    ),
    "state": ModalityConfig(
        # Single reference state for REL_XYZ_ROT6D conversion.
        delta_indices=[0],
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        # Use mean/std normalization for continuous pose and gripper values.
        mean_std_embedding_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
    ),
    "action": ModalityConfig(
        # Use consecutive indices [0..49] so horizon covers ~1.67s at 30 FPS.
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        action_configs=[
            # PSM1 pose: REL_XYZ_ROT6D EEF action (xyz + rot6d output).
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm1_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            # PSM1 gripper: absolute jaw angle.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
            # PSM2 pose: REL_XYZ_ROT6D EEF action (xyz + rot6d output).
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            # PSM2 gripper: absolute jaw angle.
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
        # "task" reads from tasks.jsonl via task_index.
        modality_keys=["task"],
    ),
}

# Register with the Obuda dVRK embodiment tag.
register_modality_config(obuda_config, embodiment_tag=EmbodimentTag.OBUDA_DVRK)
