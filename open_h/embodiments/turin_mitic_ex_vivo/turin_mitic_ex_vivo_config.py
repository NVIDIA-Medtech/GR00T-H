"""
Turin MITIC ex vivo modality configuration for GR00T N1.6.

This configuration supports the Turin MITIC ex vivo dataset with:
- Dual-arm dVRK (PSM1/PSM2)
- REL_XYZ_ROT6D EEF action representation (xyz + quaternion input)
- Joint-angle state embeddings (12D) with pass-through EEF pose references
- Stereo endoscope video (left/right)

Data format:
- State: 12D joint angles (6 per arm)
- Action: 14D absolute EEF pose (xyz + quat) for PSM1 + PSM2

REL_XYZ_ROT6D conversion:
- Reference poses are taken from action at t=0 (pass-through in state)
- Actions are predicted for t+1..t+50 at 30 Hz (~1.67s horizon)
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

turin_mitic_config = {
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
            # Embedded state keys (sent to model)
            "psm1_joints",
            "psm2_joints",
            # Pass-through keys (loaded but not embedded)
            "psm1_pose",
            "psm2_pose",
        ],
        mean_std_embedding_keys=[
            "psm1_joints",
            "psm2_joints",
        ],
        pass_through_keys=[
            "psm1_pose",
            "psm2_pose",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(1, ACTION_HORIZON + 1)),
        modality_keys=[
            "psm1_pose",
            "psm2_pose",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm1_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.instruction"],
    ),
}

# Register with the Turin MITIC ex vivo embodiment tag.
register_modality_config(turin_mitic_config, embodiment_tag=EmbodimentTag.TURIN_MITIC_EX_VIVO)
