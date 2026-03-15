"""
JHU LSCR (Laboratory for Surgical & Computational Robotics) modality configs.

This module defines built-in LSCR configurations that are auto-registered when
`open_h.embodiments` is imported. These cover the LSCR schemas that are not
already covered by the standard dVRK embodiment:
- MIRACLE (15 Hz): cartesian EEF actions (REL_XYZ_ROT6D)
- SMARTS (10 Hz): cartesian EEF actions (REL_XYZ_ROT6D)

ARCADE (30 Hz) is handled by the standard dVRK configuration, with a modality
mapping that mirrors the dVRK annotation schema.
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


# Action horizons tuned to dataset FPS.
ACTION_HORIZON_15HZ = 25
ACTION_HORIZON_10HZ = 16


jhu_lscr_miracle_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "camera_left",
            # "camera_right", # Mono Only
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
        # Cartesian pose actions are derived from the same EEF pose state, so
        # the action horizon starts at t+1 relative to the reference pose at t.
        delta_indices=list(range(1, ACTION_HORIZON_15HZ + 1)),
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm1_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="xyzw",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="xyzw",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
            ),
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
        modality_keys=["annotation.task"],
    ),
}


jhu_lscr_smarts_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope_left",
            # "endoscope_right", # Mono Only
            "camera_side_view",
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
        delta_indices=list(range(1, ACTION_HORIZON_10HZ + 1)),
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm1_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="xyzw",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="xyzw",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
            ),
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
        modality_keys=["annotation.task"],
    ),
}


register_modality_config(
    jhu_lscr_miracle_config, embodiment_tag=EmbodimentTag.JHU_LSCR_DVRK_MIRACLE
)
register_modality_config(jhu_lscr_smarts_config, embodiment_tag=EmbodimentTag.JHU_LSCR_DVRK_SMARTS)
