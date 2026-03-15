from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag


class MessageType(Enum):
    START_OF_EPISODE = "start_of_episode"
    END_OF_EPISODE = "end_of_episode"
    EPISODE_STEP = "episode_step"
    IMAGE = "image"
    TEXT = "text"


class ActionRepresentation(Enum):
    """
    Defines how action values relate to the current robot state.

    - RELATIVE: Actions are deltas from current state (computed at training time)
    - DELTA: Incremental changes per timestep
    - ABSOLUTE: Target positions with no state dependency
    - REL_XYZ_ROT6D: Translation and rotation is relative to current EEF,
                     gripper is absolute. Used for healthcare/manipulation tasks.
    """

    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"
    REL_XYZ_ROT6D = "rel_xyz_rot6d"


class ActionType(Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    """
    Defines the format of action data components.

    - DEFAULT: Default format, no specific interpretation
    - XYZ: Translation only (3D position)
    - ROT6D: 6D rotation representation (first two columns of rotation matrix)
    - XYZ_ROT6D: Combined translation and 6D rotation
    - XYZ_ROTVEC: Combined translation and rotation vector
    """

    DEFAULT = "default"
    XYZ = "xyz"
    ROT6D = "rot6d"
    XYZ_ROT6D = "xyz+rot6d"
    XYZ_ROTVEC = "xyz+rotvec"


@dataclass
class VLAStepData:
    """
    Represents a single step of VLA (Vision-Language-Action) data.

    This is the core data structure returned by datasets, containing raw observation
    and action data that will be processed by the SequenceVLAProcessor.
    """

    # Core data
    images: dict[str, list[np.ndarray]]  # view_name -> list[np.ndarray] (for temporal stacking)
    states: dict[
        str, np.ndarray
    ]  # state_name -> np.ndarray (dim,) for single step or (horizon, dim) for trajectory
    actions: dict[str, np.ndarray]  # action_name -> np.ndarray (horizon, dim) for action chunk
    masks: dict[str, list[np.ndarray]] | None = None  # view_name -> list[np.ndarray] (H, W)
    text: str | None = None  # Optional task description or instruction
    embodiment: EmbodimentTag = (
        EmbodimentTag.NEW_EMBODIMENT
    )  # Optional embodiment tag for cross-embodiment training
    is_demonstration: bool = False  # Whether the step is a demonstration. If True, no loss should be computed for this step.

    # Flexible metadata that can be extended by users
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionConfig:
    """
    Configuration for an action modality defining representation, type, format, and normalization.

    This config controls how action data is processed during training and inference,
    including conversion to relative/rel-xyz-rot6d representations and normalization.

    Attributes:
        rep: How action values relate to robot state (RELATIVE, ABSOLUTE, REL_XYZ_ROT6D, DELTA)
        type: Whether this is end-effector control (EEF) or joint control (NON_EEF)
        format: The format of the action data (XYZ, ROT6D, XYZ_ROT6D, etc.)
        state_key: Which state key to use as reference for relative actions (e.g., "eef_pose")
        input_rotation_format: Format of incoming rotation data (actions):
            - "quat": Quaternion format - will be converted to rot6d for REL_XYZ_ROT6D
            - "rot6d": Already in 6D rotation format - no conversion needed
        input_quat_order: Quaternion component ordering when input_rotation_format="quat":
            - "xyzw": Scalar-last order (x, y, z, w) - scipy convention, default
            - "wxyz": Scalar-first order (w, x, y, z) - used by some datasets (e.g., Hamlyn)
        reference_rotation_format: Format of rotation in the reference state (for REL_XYZ_ROT6D):
            - "quat": Quaternion format - state is 7D: xyz + quat
            - "rot6d": 6D rotation format - state is 9D: xyz + rot6d
        reference_quat_order: Quaternion component ordering when reference_rotation_format="quat":
            - "xyzw": Scalar-last order (default)
            - "wxyz": Scalar-first order
        translation_scaling_key: Optional state key containing translation scaling factor.
            If provided, rel-xyz-rot6d translation is multiplied by this scaling factor
            to convert from hand-controller-space to instrument-space. Used for CMR Versius.
        rotation_scaling_key: Optional state key containing rotation scaling factor.
            If provided, rel-xyz-rot6d rotation angle is multiplied by this scaling factor.
            Uses axis-angle representation for proper rotation scaling.
        hold_through_clutch: Whether this ABSOLUTE action should hold its value during clutch-out
            (sample-and-hold) instead of being zeroed (fail-safe). Only applies to ABSOLUTE actions.
            - True: Hold last engaged value (e.g., gripper holding a needle)
            - False: Zero during clutch-out (e.g., energy button - fail-safe)
            Default is False for safety - actions zero unless explicitly marked to hold.
            For RELATIVE/REL_XYZ_ROT6D actions, this flag is ignored (always zeroed).
        normalization_type: Which normalization strategy to use for this action group.
            - "temporal_meanstd": Use meanstd normalization with temporal-aware stats (default).
              Stats have shape (horizon, dim) so each timestep in the action chunk
              is normalized independently, accounting for cumulative magnitude growth
              in relative action representations like REL_XYZ_ROT6D.
            - "meanstd": Use meanstd normalization (same as temporal_meanstd)
            - "minmax": Use min/max normalization
            - "skip": Skip normalization entirely (pass through raw values)
    """

    rep: ActionRepresentation
    type: ActionType
    format: ActionFormat
    state_key: str | None = None
    input_rotation_format: str = "quat"
    input_quat_order: str = "xyzw"  # "xyzw" (scipy default) or "wxyz" (scalar-first)
    reference_rotation_format: str = "rot6d"
    reference_quat_order: str = "xyzw"  # "xyzw" (scipy default) or "wxyz" (scalar-first)
    # Motion scaling keys for CMR Versius (optional, default None = no scaling)
    translation_scaling_key: str | None = None
    rotation_scaling_key: str | None = None
    # Clutch-aware behavior for ABSOLUTE actions (default False = fail-safe zeroing)
    hold_through_clutch: bool = False
    # Normalization strategy for this action group
    normalization_type: str = "temporal_meanstd"


@dataclass
class ModalityConfig:
    """Configuration for a modality defining how data should be sampled and loaded.

    This class specifies which indices to sample relative to a base index and which
    keys to load for a particular modality (e.g., video, state, action).
    """

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""
    sin_cos_embedding_keys: list[str] | None = None
    """Optional list of keys to apply sin/cos encoding. If None or empty, use min/max normalization for all keys."""
    mean_std_embedding_keys: list[str] | None = None
    """Optional list of keys to apply mean/std normalization. If None or empty, use min/max normalization for all keys."""
    min_max_embedding_keys: list[str] | None = None
    """Optional list of keys to apply min/max normalization. If None or empty, keys not in sin_cos or mean_std will use min/max."""
    pass_through_keys: list[str] | None = None
    """Optional list of keys that are used for intermediate calculations, but ARE NOT sent to the model. Used for auxiliary data like motion scaling factors that are needed for action processing but should not be normalized."""
    action_configs: list[ActionConfig] | None = None

    def __post_init__(self):
        """Parse action configs from dictionaries if provided as dicts.

        Converts dictionary-based action configs (e.g., from JSON) into ActionConfig
        dataclass instances, handling enum parsing and default values.
        """
        if self.action_configs is not None:
            assert len(self.action_configs) == len(self.modality_keys), (
                f"Number of action configs ({len(self.action_configs)}) must match number of modality keys ({len(self.modality_keys)})"
            )
            parsed_action_configs = []
            for action_config in self.action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        rep=ActionRepresentation[action_config["rep"]],
                        type=ActionType[action_config["type"]],
                        format=ActionFormat[action_config["format"]],
                        state_key=action_config.get("state_key", None),
                        input_rotation_format=action_config.get("input_rotation_format", "quat"),
                        input_quat_order=action_config.get("input_quat_order", "xyzw"),
                        reference_rotation_format=action_config.get(
                            "reference_rotation_format", "rot6d"
                        ),
                        reference_quat_order=action_config.get("reference_quat_order", "xyzw"),
                        translation_scaling_key=action_config.get("translation_scaling_key", None),
                        rotation_scaling_key=action_config.get("rotation_scaling_key", None),
                        hold_through_clutch=action_config.get("hold_through_clutch", False),
                        normalization_type=action_config.get(
                            "normalization_type", "temporal_meanstd"
                        ),
                    )
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs


# =============================================================================
# Dataset-specific runtime behavior flags
# =============================================================================
# These flags gate behavior that should only apply to specific embodiments.
EMBODIMENTS_SKIP_NEXT_DONE: set[EmbodimentTag] = {EmbodimentTag.UCSD_DVRK}
"""
Embodiments that should skip terminal steps where `next.done == True`.

These datasets include terminal padding where the final action is zeroed out
(including zero-norm quaternions). We exclude those steps from training and
stats to avoid invalid rotations and terminal snap-to-zero artifacts.
"""


# =============================================================================
# CMR (Clutch-Mechanical-Robot) Data Keys
# =============================================================================
# These keys are used for CMR Versius surgical robot data to track haptic engagement.
# When the surgeon "clutches out" (disengages the haptic controller), the robot should
# not move, and action targets should be zeroed (for RELATIVE/REL_XYZ_ROT6D) or
# held at the last engaged value (for ABSOLUTE actions with hold_through_clutch=True).

CMR_ENGAGED_LEFT_KEY: str = "hapticengaged_left"
"""State/action key indicating if the left haptic controller is engaged (bool).
When False, the surgeon's left hand is "clutched out" and left arm should not move."""

CMR_ENGAGED_RIGHT_KEY: str = "hapticengaged_right"
"""State/action key indicating if the right haptic controller is engaged (bool).
When False, the surgeon's right hand is "clutched out" and right arm should not move."""

CMR_ARM_LINKED_LEFT_KEY: str = "armlinkedtohaptic_left"
"""State key indicating which robot arm (0-3) is linked to the left haptic controller.
Used for arm swapping detection and deriving arm_left_color from arm_X_color."""

CMR_ARM_LINKED_RIGHT_KEY: str = "armlinkedtohaptic_right"
"""State key indicating which robot arm (0-3) is linked to the right haptic controller.
Used for arm swapping detection and deriving arm_right_color from arm_X_color."""

# Arm color keys (indices 0-3 correspond to physical robot arms)
CMR_ARM_COLOR_KEYS: tuple[str, ...] = ("arm_0_color", "arm_1_color", "arm_2_color", "arm_3_color")
"""State keys for arm colors by physical arm index (0-3).
Used with armlinkedtohaptic_* to derive arm_left_color/arm_right_color."""

# -----------------------------------------------------------------------------
# CMR Raw Parquet Indices
# -----------------------------------------------------------------------------
# These indices refer to positions in the raw observation.state array in parquet files.
# Used for fast clutch-aware filtering with PyArrow (bypasses modality.json mapping).
# WARNING: These indices are specific to CMR Versius data format and must be updated
# if the observation.state layout changes.

CMR_RAW_INDEX_HAPTIC_ENGAGED_LEFT: int = 16
"""Index of hapticengaged_left in raw observation.state array."""

CMR_RAW_INDEX_HAPTIC_ENGAGED_RIGHT: int = 17
"""Index of hapticengaged_right in raw observation.state array."""

CMR_RAW_INDEX_ARM_LINKED_LEFT: int = 20
"""Index of armlinkedtohaptic_left in raw observation.state array."""

CMR_RAW_INDEX_ARM_LINKED_RIGHT: int = 21
"""Index of armlinkedtohaptic_right in raw observation.state array."""

CMR_RAW_MIN_STATE_LENGTH: int = 22
"""Minimum observation.state array length for CMR data (must include index 21)."""


# =============================================================================
# Action Dimension Constants
# =============================================================================
# These constants define the dimensions of different action format representations.

XYZ_DIM: int = 3
"""Dimension of XYZ translation component (3D position)."""

ROT6D_DIM: int = 6
"""Dimension of 6D rotation representation (first two columns of rotation matrix)."""

EEF_XYZ_ROT6D_DIM: int = 9
"""Total dimension for end-effector XYZ_ROT6D format: xyz (3) + rot6d (6) = 9."""

QUAT_DIM: int = 4
"""Dimension of quaternion rotation representation (xyzw order)."""

EEF_XYZ_QUAT_DIM: int = 7
"""Total dimension for end-effector XYZ_QUAT format: xyz (3) + quat (4) = 7."""

ROT6D_IDENTITY: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
"""Identity rotation in rot6d format: first two columns of the 3x3 identity matrix.
Used when zeroing out REL_XYZ_ROT6D actions for disengaged timesteps."""
