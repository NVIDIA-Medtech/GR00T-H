"""
Monocular dVRK (JHU) modality configuration for GR00T N1.6.

This configuration mirrors the standard dVRK embodiment but restricts the
video modality to a single camera view:

- endoscope_left only (no wrist cameras)
- same dual-arm REL_XYZ_ROT6D action representation
- same 16D state/action format as the standard dVRK config

Use this when training models that should rely solely on a monocular
endoscope feed while keeping the rest of the embodiment consistent.
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

jhu_dvrk_mono_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        # Monocular endoscope input only (no wrist cameras)
        modality_keys=[
            "endoscope_left",
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
        delta_indices=list(range(ACTION_HORIZON)),
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

# Register with the monocular dVRK tag for surgical robot finetuning
register_modality_config(jhu_dvrk_mono_config, embodiment_tag=EmbodimentTag.JHU_IMERSE_DVRK_MONO)
