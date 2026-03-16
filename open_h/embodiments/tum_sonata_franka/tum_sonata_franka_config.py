"""
TUM SonATA Ultrasound modality configuration for GR00T N1.6.

This configuration supports the TUM SonATA robotic ultrasound sonography dataset with:
- Franka Panda robot with ultrasound probe end-effector
- REL_XYZ_ROT6D action representation with Euler angle input (RPY -> rot6d)
- 3 camera views (TPV, wrist-mounted, ultrasound)
- Joint angle state + force/torque sensor data

Data Format:
- State: 7D joint angles + 6D force/torque = 13D embedded
         + 6D EEF pose (pass-through for REL_XYZ_ROT6D reference)
- Action: 6D absolute EEF pose (xyz + roll/pitch/yaw) -> 9D REL_XYZ_ROT6D

REL_XYZ_ROT6D Conversion:
- Position: relative to reference EEF position (delta xyz)
- Rotation: Euler angles (RPY) converted to rot6d relative to reference
- Output: 9D per timestep (xyz_rel + rot6d_rel)

Camera Views (in model input order):
1. tpv_camera: Third-person view - scene context
2. wrist_camera: Wrist-mounted view - probe positioning
3. ultrasound: Primary imaging modality - task-relevant feedback

Dataset: TUM SonATA (Computer Aided Medical Procedures Lab)
- SonATA_abdomen: 1,533 episodes, 325k frames
- SonATA_arm: 369 episodes
- SonATA_thyroid: 260 episodes
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

tum_sonata_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        # Camera order: external context first, then progressively more task-specific
        modality_keys=["tpv_camera", "wrist_camera", "ultrasound"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Single reference state for REL_XYZ_ROT6D
        modality_keys=[
            # Embedded state keys (sent to model)
            "joint_angles",  # 7D joint angles - mean_std normalized
            "force_torque",  # 6D force/torque sensor - mean_std normalized
            # Pass-through keys (loaded but not embedded)
            "eef_pose",  # 6D EEF pose from action column (for REL_XYZ_ROT6D reference)
        ],
        # Mean-std normalization for continuous values
        mean_std_embedding_keys=[
            "joint_angles",
            "force_torque",
        ],
        # Pass-through keys: loaded for REL_XYZ_ROT6D, never embedded to model
        # eef_pose is extracted from action column (via modality.json original_key)
        pass_through_keys=[
            "eef_pose",
        ],
    ),
    "action": ModalityConfig(
        # Start at 1 (index 0 is state reference), go to ACTION_HORIZON
        delta_indices=list(range(1, ACTION_HORIZON + 1)),
        modality_keys=["eef_pose"],  # 6D: xyz + roll/pitch/yaw (Euler)
        action_configs=[
            # EEF pose: REL_XYZ_ROT6D action with Euler input
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,  # Output: xyz_rel + rot6d_rel = 9D
                state_key="eef_pose",  # Reference from action at t=0 (via pass_through)
                normalization_type="temporal_meanstd",
                input_rotation_format="euler",  # Euler angles (roll, pitch, yaw)
                reference_rotation_format="euler",  # Reference is also Euler
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],  # Maps to tasks.jsonl (287 unique task descriptions)
    ),
}

# Register with TUM_SONATA_FRANKA tag for ultrasound robotic sonography
register_modality_config(tum_sonata_config, embodiment_tag=EmbodimentTag.TUM_SONATA_FRANKA)
