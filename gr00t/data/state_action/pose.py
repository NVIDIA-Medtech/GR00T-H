from __future__ import annotations

from enum import Enum
from typing import Optional, TypeVar, Union

from gr00t.data.types import ActionFormat
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


# TypeVar for self-type preservation in Pose operations
PoseT = TypeVar("PoseT", bound="Pose")


# =============================================================================
# Standalone rotation helper functions
# =============================================================================


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.

    The 6D representation consists of the first two columns of the rotation matrix,
    flattened. This representation is continuous and avoids gimbal lock issues.

    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"

    Args:
        rot6d: 6D rotation vector of shape (6,), representing first two columns
               of rotation matrix flattened as [r00, r10, r20, r01, r11, r21]

    Returns:
        3x3 rotation matrix

    Example:
        >>> rot6d = np.array([1, 0, 0, 0, 1, 0])  # Identity
        >>> R = rot6d_to_rotation_matrix(rot6d)
        >>> R.shape
        (3, 3)
    """
    # Reshape to (2, 3) then transpose to get columns as (3, 2)
    rot6d = np.asarray(rot6d).reshape(2, 3).T

    # First two columns of the rotation matrix
    col1 = rot6d[:, 0]
    col2 = rot6d[:, 1]

    # Normalize first column
    col1 = col1 / np.linalg.norm(col1)

    # Gram-Schmidt orthogonalization for second column
    col2 = col2 - np.dot(col1, col2) * col1
    col2 = col2 / np.linalg.norm(col2)

    # Third column is cross product (ensures right-handed coordinate system)
    col3 = np.cross(col1, col2)

    # Construct rotation matrix by stacking columns
    rotation_matrix = np.column_stack([col1, col2, col3])

    return rotation_matrix


def rotation_matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.

    Extracts the first two columns of the rotation matrix and flattens them.
    This follows the convention from Zhou et al., "On the Continuity of
    Rotation Representations in Neural Networks".

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        6D rotation vector of shape (6,), representing first two columns
        flattened as [r00, r10, r20, r01, r11, r21]

    Example:
        >>> R = np.eye(3)
        >>> rot6d = rotation_matrix_to_rot6d(R)
        >>> rot6d
        array([1., 0., 0., 0., 1., 0.])
    """
    # Extract first two columns and flatten: [:, :2] gives (3, 2), .T gives (2, 3), flatten gives (6,)
    return rotation_matrix[:, :2].T.flatten()


def rotation_matrices_to_rot6d(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Batch convert rotation matrices to 6D rotation representations.

    Extracts the first two columns of each rotation matrix and flattens them.
    This follows the convention from Zhou et al., "On the Continuity of
    Rotation Representations in Neural Networks".

    Args:
        rotation_matrices: Rotation matrices of shape (N, 3, 3)

    Returns:
        6D rotation vectors of shape (N, 6), where each vector contains
        first two columns flattened as [r00, r10, r20, r01, r11, r21]
    """
    # Extract first two columns: (N, 3, 3) -> (N, 3, 2)
    # Transpose to (N, 2, 3) then reshape to (N, 6)
    return rotation_matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def rot6ds_to_rotation_matrices(rot6ds: np.ndarray) -> np.ndarray:
    """
    Batch convert 6D rotation representations to rotation matrices.

    Uses Gram-Schmidt orthogonalization to ensure valid rotation matrices.
    The rot6d format stores the first two columns of the rotation matrix,
    following the convention from Zhou et al., "On the Continuity of
    Rotation Representations in Neural Networks".

    Args:
        rot6ds: 6D rotation vectors of shape (N, 6), where each vector contains
                the first two columns of a rotation matrix flattened as
                [r00, r10, r20, r01, r11, r21]

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    N = rot6ds.shape[0]

    # Reshape to (N, 2, 3) then transpose to get columns as (N, 3, 2)
    cols = rot6ds.reshape(N, 2, 3).transpose(0, 2, 1)

    col1 = cols[:, :, 0]  # (N, 3)
    col2 = cols[:, :, 1]  # (N, 3)

    # Gram-Schmidt orthogonalization (vectorized)
    col1_norm = np.linalg.norm(col1, axis=1, keepdims=True)
    col1 = col1 / np.maximum(col1_norm, 1e-8)

    # col2 = col2 - (col1 · col2) * col1
    dot = np.sum(col1 * col2, axis=1, keepdims=True)
    col2 = col2 - dot * col1
    col2_norm = np.linalg.norm(col2, axis=1, keepdims=True)
    col2 = col2 / np.maximum(col2_norm, 1e-8)

    # col3 = col1 × col2
    col3 = np.cross(col1, col2)

    # Stack columns to form rotation matrices: (N, 3, 3)
    return np.stack([col1, col2, col3], axis=2)


def quats_to_rotation_matrices(quats: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """
    Batch convert quaternions to rotation matrices.

    Args:
        quats: Quaternions of shape (N, 4)
        order: Quaternion ordering - "xyzw" (scipy default) or "wxyz" (scalar-first)

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    if order.lower() == "wxyz":
        # Convert from wxyz to xyzw: [w, x, y, z] -> [x, y, z, w]
        quats = quats[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quats).as_matrix()


def rotation_matrices_to_quats(rotation_matrices: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """
    Batch convert rotation matrices to quaternions.

    Args:
        rotation_matrices: Rotation matrices of shape (N, 3, 3)
        order: Desired quaternion ordering - "xyzw" or "wxyz"

    Returns:
        Quaternions of shape (N, 4)
    """
    quats_xyzw = Rotation.from_matrix(rotation_matrices).as_quat()
    if order.lower() == "wxyz":
        # Convert from xyzw to wxyz: [x, y, z, w] -> [w, x, y, z]
        return quats_xyzw[:, [3, 0, 1, 2]]
    return quats_xyzw


def eulers_to_rotation_matrices(eulers: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """
    Batch convert Euler angles to rotation matrices.

    Args:
        eulers: Euler angles of shape (N, 3) in radians (roll, pitch, yaw)
        seq: Euler angle convention - "xyz" for extrinsic rotations (default)
             This means: rotate about fixed X axis (roll), then Y (pitch), then Z (yaw)

    Returns:
        Rotation matrices of shape (N, 3, 3)

    Note:
        Euler angles can have discontinuities at ±π (wraparound), but this conversion
        handles them correctly because rotation matrices are continuous representations.
        This is particularly important for REL_XYZ_ROT6D action conversion where
        consecutive frames may have Euler angle jumps due to wraparound.
    """
    return Rotation.from_euler(seq, eulers).as_matrix()


def rotation_matrices_to_eulers(rotation_matrices: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """
    Batch convert rotation matrices to Euler angles.

    Args:
        rotation_matrices: Rotation matrices of shape (N, 3, 3)
        seq: Euler angle convention - "xyz" for extrinsic rotations (default)

    Returns:
        Euler angles of shape (N, 3) in radians (roll, pitch, yaw)

    Warning:
        Euler angles have discontinuities at ±π. For continuous action representation,
        prefer rot6d format which is always continuous.
    """
    return Rotation.from_matrix(rotation_matrices).as_euler(seq)


def quat_to_rotation_matrix(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        quat: Quaternion array of shape (4,)
        order: Quaternion ordering - "xyzw" (scipy default) or "wxyz" (scalar-first)

    Returns:
        3x3 rotation matrix

    Example:
        >>> quat = np.array([0, 0, 0, 1])  # Identity in xyzw
        >>> R = quat_to_rotation_matrix(quat, order="xyzw")
        >>> np.allclose(R, np.eye(3))
        True
    """
    quat = np.asarray(quat)
    if order.lower() == "wxyz":
        # Convert from wxyz to xyzw (scipy uses xyzw)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    return Rotation.from_quat(quat).as_matrix()


def rotation_matrix_to_quat(rotation_matrix: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion.

    Args:
        rotation_matrix: 3x3 rotation matrix
        order: Desired quaternion ordering - "xyzw" or "wxyz"

    Returns:
        Quaternion array of shape (4,)

    Example:
        >>> R = np.eye(3)
        >>> quat = rotation_matrix_to_quat(R, order="wxyz")
        >>> quat
        array([1., 0., 0., 0.])
    """
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    if order.lower() == "wxyz":
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_xyzw


def convert_to_rel_xyz_rot6d(
    action_data: np.ndarray,
    eef_pose: np.ndarray,
    input_rotation_format: str = "quat",
    reference_rotation_format: str = "rot6d",
    input_quat_order: str = "xyzw",
    reference_quat_order: str = "xyzw",
) -> np.ndarray:
    """
    Convert absolute action data to rel-xyz-rot6d representation.

    REL_XYZ_ROT6D means:
    - Translation: relative to reference EEF position (delta from reference)
    - Rotation: relative to reference orientation, expressed in 6D format
    - Gripper: absolute (unchanged, handle gripper separately)

    This representation is useful for manipulation tasks where:
    - Actions are predicted relative to a single reference state (the current observation)
    - Position is a delta from the reference EEF position
    - Rotation is relative to the reference EEF orientation

    Args:
        action_data: Absolute action data of shape (H, D) where:
            - H is the action horizon
            - D = 3 (xyz) + 4 (quat) or D = 3 (xyz) + 6 (rot6d)
            - Does NOT include gripper; handle gripper separately
        eef_pose: Reference end-effector pose:
            - Shape (9,) for rot6d format: xyz + rot6d
            - Shape (7,) for quat format: xyz + quaternion
            This is typically the current EEF pose from the observation.
        input_rotation_format: Format of rotation in action_data:
            - "quat": quaternion format
            - "rot6d": 6D rotation representation
        reference_rotation_format: Format of rotation in eef_pose:
            - "quat": quaternion format (7D pose: xyz + quat)
            - "rot6d": 6D rotation (9D pose: xyz + rot6d)
        input_quat_order: Quaternion ordering for action_data when input_rotation_format="quat":
            - "xyzw": Scalar-last (scipy convention, default)
            - "wxyz": Scalar-first (e.g., Hamlyn dataset)
        reference_quat_order: Quaternion ordering for eef_pose when reference_rotation_format="quat":
            - "xyzw": Scalar-last (default)
            - "wxyz": Scalar-first

    Returns:
        REL_XYZ_ROT6D actions of shape (H, 9) with xyz (relative) + rot6d (relative)

    Example:
        >>> action_data = np.random.randn(16, 7)  # 16 steps, xyz + quat
        >>> eef_pose = np.random.randn(7)  # xyz + quat (xyzw)
        >>> rel_xyz_rot6d = convert_to_rel_xyz_rot6d(
        ...     action_data,
        ...     eef_pose,
        ...     input_rotation_format="quat",
        ...     reference_rotation_format="quat",
        ... )
        >>> rel_xyz_rot6d.shape
        (16, 9)
    """
    H, D = action_data.shape

    # Validate input dimensions
    expected_dims = {"quat": 7, "rot6d": 9, "euler": 6}  # xyz + rotation
    if D not in expected_dims.values():
        raise ValueError(
            f"Unexpected action dimension {D}. Expected 6 (xyz+euler), 7 (xyz+quat), or 9 (xyz+rot6d)"
        )

    # Extract reference position and rotation
    ref_xyz = eef_pose[:3]

    # Parse reference rotation based on format
    if reference_rotation_format == "quat":
        ref_quat = eef_pose[3:7]
        ref_R = quat_to_rotation_matrix(ref_quat, order=reference_quat_order)
    elif reference_rotation_format == "rot6d":
        ref_rot6d = eef_pose[3:9]
        ref_R = rot6d_to_rotation_matrix(ref_rot6d)
    elif reference_rotation_format == "euler":
        # Euler angles in RPY order (roll, pitch, yaw) - radians
        # Using 'xyz' extrinsic convention (standard robotics convention)
        ref_euler = eef_pose[3:6]  # (3,)
        ref_R = Rotation.from_euler("xyz", ref_euler).as_matrix()
    else:
        raise ValueError(f"Unknown reference_rotation_format: {reference_rotation_format}")

    result = np.zeros((H, 9), dtype=np.float32)  # Always output xyz + rot6d

    # Translation: vectorized subtraction (H, 3) - (3,) broadcasts correctly
    result[:, :3] = action_data[:, :3] - ref_xyz

    # Rotation: batch convert to rotation matrices
    if input_rotation_format == "quat":
        action_quats = action_data[:, 3:7]  # (H, 4)
        action_Rs = quats_to_rotation_matrices(action_quats, order=input_quat_order)  # (H, 3, 3)
    elif input_rotation_format == "rot6d":
        action_rot6ds = action_data[:, 3:9]  # (H, 6)
        action_Rs = rot6ds_to_rotation_matrices(action_rot6ds)  # (H, 3, 3)
    elif input_rotation_format == "euler":
        # Euler angles in RPY order (roll, pitch, yaw) - radians
        # Using 'xyz' extrinsic convention (standard robotics convention)
        # This handles Euler wraparound correctly by going through rotation matrices
        action_eulers = action_data[:, 3:6]  # (H, 3)
        action_Rs = eulers_to_rotation_matrices(action_eulers, seq="xyz")  # (H, 3, 3)
    else:
        raise ValueError(f"Unknown input_rotation_format: {input_rotation_format}")

    # Relative rotation: R_ref^T @ R_action for all H matrices
    # ref_R.T is (3, 3), action_Rs is (H, 3, 3)
    # Use einsum for batch matrix multiply: (3, 3) @ (H, 3, 3) -> (H, 3, 3)
    relative_Rs = np.einsum("ij,hjk->hik", ref_R.T, action_Rs)

    # Convert back to rot6d (batch)
    result[:, 3:9] = rotation_matrices_to_rot6d(relative_Rs)

    return result


def convert_to_rel_xyz_rot6d_with_engagement(
    action_data: np.ndarray,
    eef_pose: np.ndarray,
    engaged: np.ndarray,
    input_rotation_format: str = "quat",
    reference_rotation_format: str = "quat",
    ref_engaged: bool = True,
    input_quat_order: str = "xyzw",
    reference_quat_order: str = "xyzw",
) -> np.ndarray:
    """
    Compute rel-xyz-rot6d actions with engagement-aware delta re-integration.

    Instead of computing: action[t] = pose[t] - pose[ref]
    This function computes: action[t] = sum(delta[i] * engaged[i] for i in ref+1..t)

    This correctly handles clutch scenarios in CMR Versius surgical robot data:
    - Reference disengaged → later engaged (no phantom jump from repositioning)
    - Mid-horizon clutch events (disengaged deltas zeroed)
    - Repositioning during clutch-out (not counted as arm motion)

    The key insight is that controller movement during disengaged periods doesn't
    represent actual arm motion, so we should not include those deltas in the
    cumulative relative action.

    Args:
        action_data: Absolute action data of shape (T, D) where:
            - T is the action horizon
            - D = 3 (xyz) + 4 (quat) or D = 3 (xyz) + 6 (rot6d)
            - Does NOT include gripper; handle gripper separately
        eef_pose: Reference end-effector pose:
            - Shape (7,) for quat format: xyz + quaternion
            - Shape (9,) for rot6d format: xyz + rot6d
            This is typically the current EEF pose from the observation (t=0).
        engaged: Boolean engagement mask of shape (T,)
            True where the surgeon is engaged (controlling the arm)
            False where disengaged (clutched out / menu navigation)
        input_rotation_format: Format of rotation in action_data:
            - "quat": quaternion format
            - "rot6d": 6D rotation representation
        reference_rotation_format: Format of rotation in eef_pose:
            - "quat": quaternion format (7D pose: xyz + quat)
            - "rot6d": 6D rotation (9D pose: xyz + rot6d)
        ref_engaged: Whether the reference frame (t=0 state) is engaged.
            If False, the first delta (action[0] - eef_pose) is invalid and
            will be masked out. This prevents "phantom jumps" from controller
            repositioning while clutched out. Default is True for backward
            compatibility, but should be set explicitly for CMR data.
        input_quat_order: Quaternion ordering for action_data when input_rotation_format="quat":
            - "xyzw": Scalar-last (scipy convention, default)
            - "wxyz": Scalar-first (e.g., Hamlyn dataset)
        reference_quat_order: Quaternion ordering for eef_pose when reference_rotation_format="quat":
            - "xyzw": Scalar-last (default)
            - "wxyz": Scalar-first

    Returns:
        REL_XYZ_ROT6D actions of shape (T, 9) with xyz (relative) + rot6d (relative)

    Example:
        >>> action_data = np.random.randn(16, 7)  # 16 steps, xyz + quat
        >>> eef_pose = np.random.randn(7)  # xyz + quat (xyzw)
        >>> engaged = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
        >>> rel_xyz_rot6d = convert_to_rel_xyz_rot6d_with_engagement(
        ...     action_data,
        ...     eef_pose,
        ...     engaged,
        ...     input_rotation_format="quat",
        ...     reference_rotation_format="quat",
        ...     ref_engaged=True,
        ... )
        >>> rel_xyz_rot6d.shape
        (16, 9)
    """
    T = action_data.shape[0]
    result = np.zeros((T, 9), dtype=np.float32)  # xyz(3) + rot6d(6)

    # =========================================================================
    # STEP 1: Parse reference pose (single operation, not in loop)
    # =========================================================================
    ref_xyz = eef_pose[:3]

    if reference_rotation_format == "quat":
        ref_R = quat_to_rotation_matrix(eef_pose[3:7], order=reference_quat_order)
    elif reference_rotation_format == "rot6d":
        ref_R = rot6d_to_rotation_matrix(eef_pose[3:9])
    elif reference_rotation_format == "euler":
        # Euler angles in RPY order (roll, pitch, yaw) - radians
        ref_R = Rotation.from_euler("xyz", eef_pose[3:6]).as_matrix()
    else:
        raise ValueError(f"Unknown reference_rotation_format: {reference_rotation_format}")

    # =========================================================================
    # STEP 2: Batch convert all action rotations to matrices (vectorized)
    # =========================================================================
    if input_rotation_format == "quat":
        action_Rs = quats_to_rotation_matrices(
            action_data[:, 3:7], order=input_quat_order
        )  # (T, 3, 3)
    elif input_rotation_format == "rot6d":
        action_Rs = rot6ds_to_rotation_matrices(action_data[:, 3:9])  # (T, 3, 3)
    elif input_rotation_format == "euler":
        # Euler angles in RPY order (roll, pitch, yaw) - radians
        # Handles wraparound correctly by going through rotation matrices
        action_Rs = eulers_to_rotation_matrices(action_data[:, 3:6], seq="xyz")  # (T, 3, 3)
    else:
        raise ValueError(f"Unknown input_rotation_format: {input_rotation_format}")

    # =========================================================================
    # STEP 3: Build validity mask (vectorized)
    # Delta[t] is valid iff both endpoints are engaged:
    #   - t=0: ref_engaged AND engaged[0]
    #   - t>0: engaged[t-1] AND engaged[t]
    # =========================================================================
    engaged_bool = engaged.astype(bool)
    prev_engaged = np.concatenate([[ref_engaged], engaged_bool[:-1]])  # (T,)
    delta_valid = prev_engaged & engaged_bool  # (T,)

    # =========================================================================
    # STEP 4: Compute translation deltas and cumulative sum (fully vectorized)
    # =========================================================================
    action_xyz = action_data[:, :3]  # (T, 3)

    # Prepend reference xyz to compute deltas: delta[t] = xyz[t] - xyz[t-1]
    all_xyz = np.vstack([ref_xyz[np.newaxis, :], action_xyz])  # (T+1, 3)
    delta_xyz = np.diff(all_xyz, axis=0)  # (T, 3)

    # Mask invalid deltas (set to zero)
    masked_delta_xyz = delta_xyz * delta_valid[:, np.newaxis]  # (T, 3)

    # Cumulative sum gives relative translation at each timestep
    result[:, :3] = np.cumsum(masked_delta_xyz, axis=0)

    # =========================================================================
    # STEP 5: Compute rotation deltas (vectorized batch matrix multiply)
    # delta_R[t] = prev_R[t].T @ curr_R[t]
    # =========================================================================
    # Build array of previous rotations: [ref_R, action_Rs[0], ..., action_Rs[T-2]]
    prev_Rs = np.concatenate([ref_R[np.newaxis, :, :], action_Rs[:-1]], axis=0)  # (T, 3, 3)

    # Batch compute: delta_R[t] = prev_Rs[t].T @ action_Rs[t]
    # Using einsum: 'tji,tjk->tik' transposes first matrix and multiplies
    delta_Rs = np.einsum("tji,tjk->tik", prev_Rs, action_Rs)  # (T, 3, 3)

    # For invalid deltas, set delta_R to identity (no rotation change)
    identity = np.eye(3, dtype=np.float32)
    delta_Rs = np.where(delta_valid[:, np.newaxis, np.newaxis], delta_Rs, identity)

    # =========================================================================
    # STEP 6: Cumulative rotation product (sequential - inherently not parallelizable)
    # cumulative_R[t] = cumulative_R[t-1] @ delta_R[t]
    # This is a prefix product of matrices, which must be computed sequentially.
    # =========================================================================
    cumulative_R = np.eye(3, dtype=np.float32)
    cumulative_Rs = np.zeros((T, 3, 3), dtype=np.float32)

    for t in range(T):
        cumulative_R = cumulative_R @ delta_Rs[t]
        cumulative_Rs[t] = cumulative_R

    # =========================================================================
    # STEP 7: Batch convert cumulative rotations to rot6d (vectorized)
    # =========================================================================
    result[:, 3:9] = rotation_matrices_to_rot6d(cumulative_Rs)

    return result


def convert_from_rel_xyz_rot6d(
    rel_xyz_rot6d_data: np.ndarray,
    eef_pose: np.ndarray,
    output_rotation_format: str = "rot6d",
    reference_rotation_format: str = "rot6d",
    output_quat_order: str = "xyzw",
    reference_quat_order: str = "xyzw",
) -> np.ndarray:
    """
    Convert rel-xyz-rot6d actions back to absolute representation.

    This is the inverse of convert_to_rel_xyz_rot6d.

    Args:
        rel_xyz_rot6d_data: REL_XYZ_ROT6D actions of shape (H, 9) - xyz_rel + rot6d_rel
        eef_pose: Reference end-effector pose:
            - Shape (9,) for rot6d format: xyz + rot6d
            - Shape (7,) for quat format: xyz + quaternion
            This is typically the current EEF pose from the observation.
        output_rotation_format: Desired output rotation format:
            - "rot6d": 6D rotation (output shape: H, 9)
            - "quat": quaternion format (output shape: H, 7)
        reference_rotation_format: Format of rotation in eef_pose:
            - "quat": quaternion format (7D pose: xyz + quat)
            - "rot6d": 6D rotation (9D pose: xyz + rot6d)
        output_quat_order: Quaternion ordering for output when output_rotation_format="quat":
            - "xyzw": Scalar-last (scipy convention, default)
            - "wxyz": Scalar-first
        reference_quat_order: Quaternion ordering for eef_pose when reference_rotation_format="quat":
            - "xyzw": Scalar-last (default)
            - "wxyz": Scalar-first

    Returns:
        Absolute actions with shape (H, 9) for rot6d or (H, 7) for quat

    Example:
        >>> rel_xyz_rot6d = np.random.randn(16, 9)  # 16 steps, xyz_rel + rot6d_rel
        >>> eef_pose = np.random.randn(7)  # xyz + quat (xyzw)
        >>> absolute = convert_from_rel_xyz_rot6d(
        ...     rel_xyz_rot6d,
        ...     eef_pose,
        ...     output_rotation_format="rot6d",
        ...     reference_rotation_format="quat",
        ... )
        >>> absolute.shape
        (16, 9)
    """
    H = rel_xyz_rot6d_data.shape[0]

    # Extract reference position and rotation
    ref_xyz = eef_pose[:3]

    # Parse reference rotation based on format
    if reference_rotation_format == "quat":
        ref_quat = eef_pose[3:7]
        ref_R = quat_to_rotation_matrix(ref_quat, order=reference_quat_order)
    elif reference_rotation_format == "rot6d":
        ref_rot6d = eef_pose[3:9]
        ref_R = rot6d_to_rotation_matrix(ref_rot6d)
    elif reference_rotation_format == "euler":
        # Euler angles in RPY order (roll, pitch, yaw) - radians
        ref_euler = eef_pose[3:6]
        ref_R = Rotation.from_euler("xyz", ref_euler).as_matrix()
    else:
        raise ValueError(f"Unknown reference_rotation_format: {reference_rotation_format}")

    if output_rotation_format == "rot6d":
        result = np.zeros((H, 9), dtype=np.float32)
    elif output_rotation_format == "quat":
        result = np.zeros((H, 7), dtype=np.float32)
    elif output_rotation_format == "euler":
        result = np.zeros((H, 6), dtype=np.float32)
    else:
        raise ValueError(f"Unknown output_rotation_format: {output_rotation_format}")

    # Translation: vectorized addition (H, 3) + (3,) broadcasts correctly
    result[:, :3] = rel_xyz_rot6d_data[:, :3] + ref_xyz

    # Rotation: batch convert relative rot6d to rotation matrices
    relative_rot6ds = rel_xyz_rot6d_data[:, 3:9]  # (H, 6)
    relative_Rs = rot6ds_to_rotation_matrices(relative_rot6ds)  # (H, 3, 3)

    # Absolute rotation: R_action = R_ref @ R_relative for all H matrices
    # ref_R is (3, 3), relative_Rs is (H, 3, 3)
    # Use einsum for batch matrix multiply: (3, 3) @ (H, 3, 3) -> (H, 3, 3)
    action_Rs = np.einsum("ij,hjk->hik", ref_R, relative_Rs)

    # Convert to output format (batch)
    if output_rotation_format == "rot6d":
        result[:, 3:9] = rotation_matrices_to_rot6d(action_Rs)
    elif output_rotation_format == "quat":
        result[:, 3:7] = rotation_matrices_to_quats(action_Rs, order=output_quat_order)
    else:
        # Note: Euler output has discontinuities at ±π. For continuous action
        # representation during inference, prefer rot6d output.
        result[:, 3:6] = rotation_matrices_to_eulers(action_Rs, seq="xyz")

    return result


# =============================================================================
# Motion scaling functions for CMR Versius
# =============================================================================


def scale_rot6d_by_angle(rot6d: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale a rot6d representation by scaling its axis-angle magnitude.

    For relative rotations, this effectively scales the magnitude of the rotation
    while preserving the axis of rotation. This is used for CMR Versius motion
    scaling normalization.

    Args:
        rot6d: 6D rotation representation of shape (6,) or (N, 6)
        scale_factor: Factor to multiply the rotation angle by

    Returns:
        Scaled rot6d representation with same shape as input

    Example:
        >>> # Scale a 90-degree rotation by 0.5 to get 45-degree rotation
        >>> rot6d = np.array([1, 0, 0, 0, 1, 0])  # ~identity
        >>> scaled = scale_rot6d_by_angle(rot6d, 0.5)
    """
    # Handle batch dimension
    single_input = rot6d.ndim == 1
    if single_input:
        rot6d = rot6d[np.newaxis, :]

    # Convert rot6d to rotation matrices
    rot_matrices = rot6ds_to_rotation_matrices(rot6d)  # (N, 3, 3)

    # Convert to axis-angle via scipy Rotation
    rotations = Rotation.from_matrix(rot_matrices)
    rotvecs = rotations.as_rotvec()  # (N, 3) - axis * angle

    # Handle near-zero rotations to avoid numerical instability
    angle_magnitudes = np.linalg.norm(rotvecs, axis=-1, keepdims=True)
    epsilon = 1e-8

    # Scale the rotation vector (this scales the angle while preserving axis)
    # For very small rotations, keep them unchanged to avoid division issues
    scaled_rotvecs = np.where(
        angle_magnitudes > epsilon,
        rotvecs * scale_factor,
        rotvecs,  # Keep unchanged for near-identity rotations
    )

    # Convert back to rotation matrices
    scaled_rotations = Rotation.from_rotvec(scaled_rotvecs)
    scaled_matrices = scaled_rotations.as_matrix()  # (N, 3, 3)

    # Convert back to rot6d
    scaled_rot6d = rotation_matrices_to_rot6d(scaled_matrices)  # (N, 6)

    if single_input:
        scaled_rot6d = scaled_rot6d[0]

    return scaled_rot6d


def apply_motion_scaling_to_rel_xyz_rot6d(
    rel_xyz_rot6d_data: np.ndarray,
    translation_scaling: float,
    rotation_scaling: float,
) -> np.ndarray:
    """
    Apply motion scaling normalization to rel-xyz-rot6d actions.

    This converts from "hand-controller-space" to "instrument-space" by
    multiplying by the scaling factors. This ensures that the same visual
    outcome (instrument movement) produces the same normalized action
    regardless of the motion scaling settings used by the surgeon.

    Scaling relationship: instrument_movement = hand_movement * scaling

    Args:
        rel_xyz_rot6d_data: REL_XYZ_ROT6D actions of shape (H, 9) - xyz_rel + rot6d_rel
            These are in hand-controller-space (raw relative movements)
        translation_scaling: Translation scaling factor (e.g., 0.333, 0.5, 1.0)
        rotation_scaling: Rotation scaling factor (e.g., 1.0, 1.5, 2.0)

    Returns:
        Motion-scaled REL_XYZ_ROT6D actions of shape (H, 9) in instrument-space

    Example:
        >>> # With translationScaling=0.333, hand moves 3cm, instrument moves 1cm
        >>> # After scaling: action represents 1cm instrument movement
        >>> scaled = apply_motion_scaling_to_rel_xyz_rot6d(
        ...     rel_xyz_rot6d_data, translation_scaling=0.333, rotation_scaling=2.0
        ... )
    """
    result = np.zeros_like(rel_xyz_rot6d_data)

    # Scale translation: multiply by translationScaling to get instrument-space
    # (larger controller movement -> smaller instrument movement when scaling < 1)
    result[:, :3] = rel_xyz_rot6d_data[:, :3] * translation_scaling

    # Scale rotation: multiply by rotationScaling to get instrument-space
    # Use axis-angle scaling for proper rotation scaling
    rot6d = rel_xyz_rot6d_data[:, 3:9]
    result[:, 3:9] = scale_rot6d_by_angle(rot6d, rotation_scaling)

    return result


def unapply_motion_scaling_from_rel_xyz_rot6d(
    scaled_rel_xyz_rot6d_data: np.ndarray,
    translation_scaling: float,
    rotation_scaling: float,
) -> np.ndarray:
    """
    Reverse motion scaling to convert back to hand-controller-space.

    Used during inference to convert model predictions (in instrument-space)
    back to actual hand controller commands that the robot system expects.

    Args:
        scaled_rel_xyz_rot6d_data: Motion-scaled REL_XYZ_ROT6D actions of shape (H, 9)
            These are in instrument-space (model predictions)
        translation_scaling: Translation scaling factor
        rotation_scaling: Rotation scaling factor

    Returns:
        Original REL_XYZ_ROT6D actions of shape (H, 9) in hand-controller-space

    Example:
        >>> # Convert instrument-space prediction back to hand controller commands
        >>> hand_controller_action = unapply_motion_scaling_from_rel_xyz_rot6d(
        ...     model_prediction, translation_scaling=0.333, rotation_scaling=2.0
        ... )
    """
    result = np.zeros_like(scaled_rel_xyz_rot6d_data)

    # Divide translation by scaling factor to get hand-controller-space
    result[:, :3] = scaled_rel_xyz_rot6d_data[:, :3] / translation_scaling

    # Divide rotation by scaling factor (scale by 1/rotation_scaling)
    rot6d = scaled_rel_xyz_rot6d_data[:, 3:9]
    result[:, 3:9] = scale_rot6d_by_angle(rot6d, 1.0 / rotation_scaling)

    print("WARNING: Batch size > 1 may have problems due to global use of scaling")

    return result


def invert_transformation(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Invert a homogeneous transformation matrix.

    Args:
        T: A 4x4 homogeneous transformation matrix

    Returns:
        The inverse of the transformation matrix (4x4)
    """
    R = T[:3, :3]  # Extract the rotation matrix
    t = T[:3, 3]  # Extract the translation vector

    # Inverse of the rotation matrix is its transpose (since it's orthogonal)
    R_inv = R.T

    # Inverse of the translation is -R_inv * t
    t_inv = -R_inv @ t

    # Construct the inverse transformation matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


def relative_transformation(
    T0: NDArray[np.float64], Tt: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the relative transformation between two poses.

    Args:
        T0: Initial 4x4 homogeneous transformation matrix
        Tt: Current 4x4 homogeneous transformation matrix

    Returns:
        The relative transformation matrix (4x4) from T0 to Tt
    """
    # Relative transformation is T0^{-1} * Tt
    T_relative = invert_transformation(T0) @ Tt
    return T_relative


class RotationType(Enum):
    """Supported rotation representation types"""

    QUAT = "quat"
    EULER = "euler"
    ROTVEC = "rotvec"
    MATRIX = "matrix"
    ROT6D = "rot6d"


class EulerOrder(Enum):
    """Common Euler angle conventions"""

    XYZ = "xyz"
    ZYX = "zyx"
    XZY = "xzy"
    YXZ = "yxz"
    YZX = "yzx"
    ZXY = "zxy"


class QuatOrder(Enum):
    """Quaternion ordering conventions"""

    WXYZ = "wxyz"  # scalar-first (w, x, y, z)
    XYZW = "xyzw"  # scalar-last (x, y, z, w)


class Pose:
    """
    Abstract base class for robot poses.

    This class provides common functionality for different pose representations
    including relative pose computation via the subtraction operator.
    """

    pose_type: str

    def __sub__(self: PoseT, other: PoseT) -> PoseT:
        """
        Compute relative transformation between two poses.

        For EndEffectorPose: Computes the relative transformation from other to self.
        Result represents the transformation needed to go from other's frame to self's frame.

        For JointPose: Computes the joint-space difference (self - other).

        Args:
            other: The reference pose to compute relative transformation from

        Returns:
            Relative pose (same type as self)

        Raises:
            TypeError: If poses are not of the same type

        Examples:
            # End-effector poses
            pose1 = EndEffectorPose(translation=[1, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
            pose2 = EndEffectorPose(translation=[2, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
            relative = pose2 - pose1  # Transformation from pose1 to pose2

            # Joint poses
            joint1 = JointPose([0.0, 0.5, 1.0])
            joint2 = JointPose([0.1, 0.6, 1.2])
            joint_diff = joint2 - joint1  # Joint differences: [0.1, 0.1, 0.2]
        """
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot compute relative transformation between different pose types: "
                f"{type(self).__name__} and {type(other).__name__}"
            )

        return self._compute_relative(other)

    def _compute_relative(self: PoseT, other: PoseT) -> PoseT:
        """
        Internal method to compute relative transformation.
        Must be implemented by subclasses.

        Args:
            other: The reference pose

        Returns:
            Relative pose
        """
        raise NotImplementedError("Subclasses must implement _compute_relative")

    def copy(self: PoseT) -> PoseT:
        """
        Create a deep copy of this pose.
        Must be implemented by subclasses.

        Returns:
            New Pose instance with copied data
        """
        raise NotImplementedError("Subclasses must implement copy")


class JointPose(Pose):
    """
    Represents a robot configuration in joint space.

    This class stores joint angles/positions for a robot manipulator.
    Unlike end-effector poses, joint poses represent the configuration
    of all joints in the kinematic chain.

    Examples:
        # Create a 6-DOF joint configuration
        joint_pose = JointPose(
            joints=[0.0, -np.pi/4, np.pi/2, 0.0, np.pi/4, 0.0],
            joint_names=["shoulder_pan", "shoulder_lift", "elbow",
                        "wrist_1", "wrist_2", "wrist_3"]
        )

        # Create with default joint names
        joint_pose = JointPose(joints=[0.0, 0.5, 1.0])

        # Get as dictionary
        joint_dict = joint_pose.to_dict()  # {"joint_0": 0.0, ...}

        # Access individual joints
        first_joint = joint_pose.joints[0]
        num_joints = joint_pose.num_joints

        # Compute relative joint displacement
        joint1 = JointPose([0.0, 0.5, 1.0])
        joint2 = JointPose([0.1, 0.6, 1.2])
        relative = joint2 - joint1  # [0.1, 0.1, 0.2]
    """

    pose_type = "joint"

    def __init__(
        self,
        joints: Union[list, np.ndarray],
        joint_names: Optional[list] = None,
    ):
        """
        Initialize a joint pose.

        Args:
            joints: Joint angles/positions as array-like of shape (n,)
            joint_names: Optional list of names for each joint. If None,
                        defaults to ["joint_0", "joint_1", ...]
        """
        super().__init__()
        self.joints = np.array(joints, dtype=np.float64)

        # Set defaults and validate joint_names
        if joint_names is None:
            self.joint_names = [f"joint_{i}" for i in range(len(self.joints))]
        else:
            if len(joint_names) != len(self.joints):
                raise ValueError(
                    f"Number of joint names ({len(joint_names)}) must match "
                    f"number of joints ({len(self.joints)})"
                )
            self.joint_names = joint_names

    @property
    def num_joints(self) -> int:
        """
        Get the number of joints.

        Returns:
            Number of joints in the configuration
        """
        return len(self.joints)

    def to_dict(self) -> dict:
        """
        Convert joint configuration to dictionary.

        Returns:
            Dictionary mapping joint names to joint values
        """
        return dict(zip(self.joint_names, self.joints))

    def _compute_relative(self, other):  # type: ignore[override]
        """
        Compute relative joint displacement.

        Args:
            other: Reference joint pose

        Returns:
            JointPose representing the joint-space difference (self - other)

        Raises:
            ValueError: If joint dimensions don't match
        """
        if len(self.joints) != len(other.joints):
            raise ValueError(
                f"Cannot compute relative joint pose: "
                f"joint dimensions don't match ({len(self.joints)} vs {len(other.joints)})"
            )

        relative_joints = self.joints - other.joints
        return JointPose(joints=relative_joints, joint_names=self.joint_names)

    def copy(self) -> JointPose:
        """
        Create a deep copy of this joint pose.

        Returns:
            New JointPose instance with copied data
        """
        return JointPose(
            joints=self.joints.copy(),
            joint_names=self.joint_names.copy(),
        )

    def __repr__(self) -> str:
        if len(self.joints) <= 6:
            joints_str = np.array2string(self.joints, precision=4, suppress_small=True)
        else:
            joints_str = (
                f"[{self.joints[0]:.4f}, ..., {self.joints[-1]:.4f}] ({len(self.joints)} joints)"
            )

        return f"JointPose(joints={joints_str})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, JointPose):
            return False
        return np.allclose(self.joints, other.joints) and self.joint_names == other.joint_names

    def __getitem__(self, index) -> Union[float, NDArray[np.float64]]:
        """Allow indexing: joint_pose[0] returns first joint value"""
        return self.joints[index]

    def __len__(self) -> int:
        """Allow len(): len(joint_pose) returns number of joints"""
        return len(self.joints)


class EndEffectorPose(Pose):
    """
    Represents a single end-effector pose with translation and rotation components.

    This class handles Cartesian space representations of robot end-effector poses,
    supporting multiple rotation representations (quaternions, Euler angles, rotation
    vectors, rotation matrices, etc.).

    Examples:
        # Create with quaternion (wxyz order)
        pose = EndEffectorPose(
            translation=[1.0, 2.0, 3.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
            rotation_type="quat",
            rotation_order="wxyz"
        )

        # Create with Euler angles (degrees by default)
        pose = EndEffectorPose(
            translation=[1, 2, 3],
            rotation=[0, 0, 90],
            rotation_type="euler",
            rotation_order="xyz"
        )

        # Create with Euler angles in radians
        pose = EndEffectorPose(
            translation=[1, 2, 3],
            rotation=[0, 0, np.pi/2],
            rotation_type="euler",
            rotation_order="xyz",
            degrees=False
        )

        # Create from homogeneous matrix
        H = np.eye(4)
        H[:3, 3] = [1, 2, 3]
        pose = EndEffectorPose(homogeneous=H)

        # Convert between representations
        quat_wxyz = pose.to_rotation("quat", "wxyz")
        euler_zyx = pose.to_rotation("euler", "zyx")
        rot6d = pose.to_rotation("rot6d")

        # Compute relative transformation
        pose1 = EndEffectorPose(translation=[1, 0, 0], rotation=[1,0,0,0],
                               rotation_type="quat", rotation_order="wxyz")
        pose2 = EndEffectorPose(translation=[2, 0, 0], rotation=[1,0,0,0],
                               rotation_type="quat", rotation_order="wxyz")
        relative = pose2 - pose1  # Transformation from pose1's frame to pose2's frame
    """

    pose_type = "end_effector"

    def __init__(
        self,
        translation: Optional[Union[list, np.ndarray]] = None,
        rotation: Optional[Union[list, np.ndarray]] = None,
        rotation_type: Optional[str] = None,
        rotation_order: Optional[str] = None,
        homogeneous: Optional[np.ndarray] = None,
        degrees: bool = True,
    ):
        """
        Initialize an end-effector pose.

        Args:
            translation: Translation vector [x, y, z]
            rotation: Rotation in specified format
            rotation_type: Type of rotation ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            homogeneous: Homogeneous transformation matrix (4, 4)
                        If provided, overrides translation and rotation
            degrees: For Euler angles, whether the input is in degrees (default True)
        """
        super().__init__()

        # Cache for homogeneous matrix
        self._homogeneous_cache: Optional[NDArray[np.float64]] = None
        self._cache_valid = False

        # Handle homogeneous matrix input
        if homogeneous is not None:
            self._from_homogeneous(homogeneous)
            return

        # Store translation
        self._translation = np.array(translation) if translation is not None else np.zeros(3)

        # Store rotation as scipy Rotation object internally
        if rotation is not None:
            if rotation_type is None:
                raise ValueError("rotation_type must be specified when rotation is provided")
            self._set_rotation(rotation, rotation_type, rotation_order, degrees)
        else:
            self._rotation = Rotation.identity()

    def _from_homogeneous(self, homogeneous: np.ndarray):
        """Initialize from homogeneous transformation matrix"""
        homogeneous = np.array(homogeneous)

        # Extract translation (last column, first 3 rows)
        self._translation = homogeneous[:3, 3]

        # Extract rotation matrix (top-left 3x3)
        rotation_matrix = homogeneous[:3, :3]

        # Create Rotation object from matrix
        self._rotation = Rotation.from_matrix(rotation_matrix)

    @staticmethod
    def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
        """
        Convert 6D rotation representation to rotation matrix.

        Delegates to module-level rot6d_to_rotation_matrix function.

        Args:
            rot6d: 6D rotation as (6,) array - first two rows of rotation matrix flattened

        Returns:
            Rotation matrix (3, 3)
        """
        return rot6d_to_rotation_matrix(rot6d)

    @staticmethod
    def _matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to 6D rotation representation.

        Delegates to module-level rotation_matrix_to_rot6d function.

        Args:
            rotation_matrix: Rotation matrix (3, 3)

        Returns:
            6D rotation - (6,) array (first two rows flattened)
        """
        return rotation_matrix_to_rot6d(rotation_matrix)

    def _set_rotation(
        self,
        rotation: Union[list, np.ndarray],
        rotation_type: str,
        rotation_order: Optional[str] = None,
        degrees: bool = True,
    ):
        """Internal method to set rotation from various representations"""
        rotation = np.array(rotation)
        rot_type = RotationType(rotation_type.lower())

        if rot_type == RotationType.QUAT:
            quat_order = QuatOrder(rotation_order.lower()) if rotation_order else QuatOrder.WXYZ
            if quat_order == QuatOrder.WXYZ:
                # scipy uses xyzw order, so convert
                quat_xyzw = np.array([rotation[1], rotation[2], rotation[3], rotation[0]])
            else:
                quat_xyzw = rotation
            self._rotation = Rotation.from_quat(quat_xyzw)

        elif rot_type == RotationType.EULER:
            euler_order = EulerOrder(rotation_order.lower()) if rotation_order else EulerOrder.XYZ
            self._rotation = Rotation.from_euler(euler_order.value, rotation, degrees=degrees)

        elif rot_type == RotationType.ROTVEC:
            self._rotation = Rotation.from_rotvec(rotation)

        elif rot_type == RotationType.MATRIX:
            self._rotation = Rotation.from_matrix(rotation)

        elif rot_type == RotationType.ROT6D:
            rotation_matrix = self._rot6d_to_matrix(rotation)
            self._rotation = Rotation.from_matrix(rotation_matrix)

        else:
            raise ValueError(f"Unknown rotation type: {rotation_type}")

        # Invalidate cache
        self._cache_valid = False

    @property
    def translation(self) -> np.ndarray:
        """
        Get translation vector.

        Returns:
            Translation array - shape (3,)
        """
        return self._translation.copy()

    @property
    def quat_wxyz(self) -> np.ndarray:
        """Get rotation as quaternion in wxyz order (w, x, y, z)"""
        return self.to_rotation("quat", "wxyz")

    @property
    def quat_xyzw(self) -> np.ndarray:
        """Get rotation as quaternion in xyzw order (x, y, z, w)"""
        return self.to_rotation("quat", "xyzw")

    @property
    def euler_xyz(self) -> np.ndarray:
        """Get rotation as Euler angles in xyz order (degrees)"""
        return self.to_rotation("euler", "xyz")

    @property
    def rotvec(self) -> np.ndarray:
        """Get rotation as rotation vector (axis-angle)"""
        return self.to_rotation("rotvec")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation as 3x3 rotation matrix"""
        return self.to_rotation("matrix")

    @property
    def rot6d(self) -> np.ndarray:
        """Get rotation as 6D representation (first two rows of rotation matrix)"""
        return self.to_rotation("rot6d")

    @property
    def xyz_rot6d(self) -> np.ndarray:
        """Get pose as concatenated translation and 6D rotation (9,)"""
        return np.concatenate([self._translation, self.rot6d])

    @property
    def xyz_rotvec(self) -> np.ndarray:
        """Get pose as concatenated translation and rotation vector (6,)"""
        return np.concatenate([self._translation, self.rotvec])

    @property
    def homogeneous(self) -> np.ndarray:
        """
        Get homogeneous transformation matrix.

        Returns:
            Homogeneous matrix - shape (4, 4)
        """
        if not self._cache_valid:
            self._homogeneous_cache = self._compute_homogeneous()
            self._cache_valid = True
        assert self._homogeneous_cache is not None
        return self._homogeneous_cache.copy()

    def _compute_homogeneous(self) -> np.ndarray:
        """Compute homogeneous transformation matrix"""
        H = np.eye(4)
        H[:3, :3] = self._rotation.as_matrix()
        H[:3, 3] = self._translation
        return H

    def to_rotation(
        self,
        rotation_type: str,
        rotation_order: Optional[str] = None,
        degrees: bool = True,
    ) -> np.ndarray:
        """
        Get rotation in specified representation.

        Args:
            rotation_type: Desired type ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            degrees: For Euler angles, return in degrees (default True)

        Returns:
            Rotation in requested format
            - Shape (4,) for quat
            - Shape (3,) for euler/rotvec
            - Shape (6,) for rot6d
            - Shape (3, 3) for matrix
        """
        rot_type = RotationType(rotation_type.lower())

        if rot_type == RotationType.ROT6D:
            rotation_matrix = self._rotation.as_matrix()
            return self._matrix_to_rot6d(rotation_matrix)

        if rot_type == RotationType.QUAT:
            quat_order = QuatOrder(rotation_order.lower()) if rotation_order else QuatOrder.WXYZ
            quat_xyzw = self._rotation.as_quat()
            if quat_order == QuatOrder.WXYZ:
                return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            else:
                return quat_xyzw

        elif rot_type == RotationType.EULER:
            euler_order = EulerOrder(rotation_order.lower()) if rotation_order else EulerOrder.XYZ
            return self._rotation.as_euler(euler_order.value, degrees=degrees)

        elif rot_type == RotationType.ROTVEC:
            return self._rotation.as_rotvec()

        elif rot_type == RotationType.MATRIX:
            return self._rotation.as_matrix()

        else:
            raise ValueError(f"Unknown rotation type: {rotation_type}")

    def to_homogeneous(self) -> np.ndarray:
        """
        Convert pose to homogeneous transformation matrix.
        (Alias for the homogeneous property)

        Returns:
            Homogeneous matrix - shape (4, 4)
        """
        return self.homogeneous

    def set_rotation(
        self,
        rotation: Union[list, np.ndarray],
        rotation_type: str,
        rotation_order: Optional[str] = None,
        degrees: bool = True,
    ):
        """
        Set rotation from specified representation.

        Args:
            rotation: Rotation data
            rotation_type: Type of rotation ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            degrees: For Euler angles, whether the input is in degrees (default True)
        """
        self._set_rotation(rotation, rotation_type, rotation_order, degrees)

    def _compute_relative(self, other):  # type: ignore[override]
        """
        Compute relative transformation from other to self.

        The result represents the transformation needed to go from other's frame to self's frame.
        Mathematically: T_relative = T_other^{-1} * T_self

        Args:
            other: Reference end-effector pose

        Returns:
            EndEffectorPose representing the relative transformation
        """
        # Get homogeneous matrices
        T_self = self.homogeneous
        T_other = other.homogeneous

        # Compute relative transformation: T_other^{-1} * T_self
        T_relative = relative_transformation(T_other, T_self)

        # Create new EndEffectorPose from relative transformation
        return EndEffectorPose(homogeneous=T_relative)

    @classmethod
    def from_action_format(cls, data: np.ndarray, action_format: ActionFormat) -> EndEffectorPose:
        """
        Create an EndEffectorPose from a flat array using the specified action format.

        This is the inverse of the xyz_rot6d / xyz_rotvec / homogeneous properties.

        Args:
            data: Flat array whose layout depends on action_format.
            action_format: One of ActionFormat.XYZ_ROT6D, XYZ_ROTVEC, or DEFAULT.

        Returns:
            EndEffectorPose instance.
        """
        if action_format == ActionFormat.XYZ_ROT6D:
            return cls(translation=data[:3], rotation=data[3:], rotation_type="rot6d")
        elif action_format == ActionFormat.XYZ_ROTVEC:
            return cls(translation=data[:3], rotation=data[3:], rotation_type="rotvec")
        elif action_format == ActionFormat.DEFAULT:
            return cls(homogeneous=data.reshape(4, 4))
        else:
            raise ValueError(f"Unsupported ActionFormat: {action_format}")

    def copy(self) -> EndEffectorPose:
        """
        Create a deep copy of this end-effector pose.

        Returns:
            New EndEffectorPose instance with copied data
        """
        return EndEffectorPose(
            translation=self._translation.copy(),
            rotation=self._rotation.as_quat(),
            rotation_type="quat",
            rotation_order="xyzw",
        )

    def __repr__(self) -> str:
        quat = self.to_rotation("quat", "wxyz")
        return f"EndEffectorPose(translation={self.translation}, rotation_quat_wxyz={quat})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EndEffectorPose):
            return False
        return np.allclose(self._translation, other._translation) and np.allclose(
            self._rotation.as_quat(), other._rotation.as_quat()
        )
