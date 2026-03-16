"""
Moon Surgical (Maestro) modality configuration for GR00T N1.6.

This configuration targets the Moon Surgical assistant dataset:
- 65 episodes, 30 Hz
- Dual-arm robot with 9 joint positions per arm
- Delta Cartesian translation actions per arm (xyz only)

Key design choices:
- Use joint positions as state
- Treat action deltas as ActionRepresentation.DELTA with ActionFormat.XYZ
- Use tasks.jsonl via annotation.task mapping
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
# 30 FPS -> 50 frames = 1.67 seconds of prediction
ACTION_HORIZON = 50

moon_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "scope",
            "topcam",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm_joints",
            "left_arm_joints",
        ],
        mean_std_embedding_keys=[
            "right_arm_joints",
            "left_arm_joints",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "right_arm_delta_xyz",
            "left_arm_delta_xyz",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.XYZ,
                normalization_type="temporal_meanstd",
            ),
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.XYZ,
                normalization_type="temporal_meanstd",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.task"],
    ),
}

# Register Moon Surgical embodiment config
register_modality_config(moon_config, embodiment_tag=EmbodimentTag.MOON_MAESTRO)
