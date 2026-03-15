"""Shared helpers for constructing CMR state-derived prompt prefixes.

This module centralizes the common "state prefix" logic used by multiple CMR
prompt-writing scripts. The prefix format is:

    arm left: <linked_val>. left instrument: <instr_name> (<color_name>).
    arm right: <linked_val>. right instrument: <instr_name> (<color_name>)

The final task/procedure suffix (for example, "do a prostatectomy" or
"do the suturing task") should be appended by the calling script.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict


# arm_0_color through arm_3_color live in observation.state[2:6]
ARM_COLOR_START_IDX = 2
ARM_COLOR_COUNT = 4

# armlinkedtohaptic_left / right
LINKED_LEFT_IDX = 20
LINKED_RIGHT_IDX = 21

# instrtype_left / right
INSTRTYPE_LEFT_IDX = 22
INSTRTYPE_RIGHT_IDX = 23


class CMRStatePromptFields(TypedDict):
    """Strongly-typed dictionary of CMR fields needed for prefix generation."""

    armlinkedtohaptic_left: int
    armlinkedtohaptic_right: int
    instrtype_left: int
    instrtype_right: int
    arm_0_color: int
    arm_1_color: int
    arm_2_color: int
    arm_3_color: int


# InstrType enum -> human-readable instrument name
INSTRTYPE_NAMES: dict[int, str] = {
    0: "Tool #0",
    1: "Tool #1",
    2: "Tool #2",
    3: "Tool #3",
    4: "Tool #4",
    5: "Tool #5",
    6: "Tool #6",
    7: "Tool #7",
    8: "Tool #8",
    9: "Tool #9",
    10: "Tool #10",
    11: "Tool #11",
    12: "Tool #12",
    13: "Tool #13",
    14: "Tool #14",
    15: "Tool #15",
    16: "Tool #16",
    17: "Tool #17",
    18: "Tool #18",
}

# Color enum -> human-readable color name
COLOR_NAMES: dict[int, str] = {
    0: "None",
    1: "Green",
    2: "Blue",
    3: "Cyan",
    4: "Orange",
    5: "Purple",
    6: "White",
    7: "Pink",
}


def instrtype_to_name(val: int) -> str:
    """Map an instrument enum value to a human-readable name.

    Args:
        val: Integer value from CMR `instrtype_*`.

    Returns:
        Human-readable instrument name. Unknown values fall back to
        ``"Instrument"``.
    """

    return INSTRTYPE_NAMES.get(val, "Instrument")


def color_to_name(val: int) -> str:
    """Map a color enum value to a human-readable color name.

    Args:
        val: Integer value from CMR arm color fields.

    Returns:
        Human-readable color name. Unknown values fall back to ``"None"``.
    """

    return COLOR_NAMES.get(val, "None")


def _derive_controller_color(linked_val: int, arm_colors: Sequence[int]) -> int:
    """Derive controller color using linkage and the four arm colors.

    Args:
        linked_val: Linkage arm slot index (0-3) or -1 for disconnected.
        arm_colors: Sequence of exactly four arm color enum values.

    Returns:
        The selected color enum value, or 0 (None) for disconnected/invalid
        linkage values.
    """

    if linked_val < 0:
        return 0
    if linked_val >= len(arm_colors):
        return 0
    return int(arm_colors[linked_val])


def extract_state_prompt_fields(state: Sequence[float]) -> CMRStatePromptFields:
    """Extract all CMR fields needed for prefix construction from state.

    Args:
        state: Full per-timestep ``observation.state`` sequence.

    Returns:
        Dictionary containing the required linked/instrument/arm-color fields.

    Raises:
        ValueError: If ``state`` does not contain enough entries for required
            indices.
    """

    min_required_len = INSTRTYPE_RIGHT_IDX + 1
    if len(state) < min_required_len:
        raise ValueError(
            f"Expected observation.state length >= {min_required_len}, got {len(state)}"
        )

    return CMRStatePromptFields(
        armlinkedtohaptic_left=int(round(state[LINKED_LEFT_IDX])),
        armlinkedtohaptic_right=int(round(state[LINKED_RIGHT_IDX])),
        instrtype_left=int(round(state[INSTRTYPE_LEFT_IDX])),
        instrtype_right=int(round(state[INSTRTYPE_RIGHT_IDX])),
        arm_0_color=int(round(state[ARM_COLOR_START_IDX + 0])),
        arm_1_color=int(round(state[ARM_COLOR_START_IDX + 1])),
        arm_2_color=int(round(state[ARM_COLOR_START_IDX + 2])),
        arm_3_color=int(round(state[ARM_COLOR_START_IDX + 3])),
    )


def build_state_prefix_from_fields(
    *,
    armlinkedtohaptic_left: int,
    armlinkedtohaptic_right: int,
    instrtype_left: int,
    instrtype_right: int,
    arm_0_color: int,
    arm_1_color: int,
    arm_2_color: int,
    arm_3_color: int,
) -> str:
    """Build the shared CMR prompt prefix from explicit state fields.

    Args:
        armlinkedtohaptic_left: Left controller linked arm slot index.
        armlinkedtohaptic_right: Right controller linked arm slot index.
        instrtype_left: Instrument enum on the left controller.
        instrtype_right: Instrument enum on the right controller.
        arm_0_color: Color enum for arm slot 0.
        arm_1_color: Color enum for arm slot 1.
        arm_2_color: Color enum for arm slot 2.
        arm_3_color: Color enum for arm slot 3.

    Returns:
        Prefix string containing left/right linkage and instrument/color details.
    """

    arm_colors = [arm_0_color, arm_1_color, arm_2_color, arm_3_color]
    left_color = _derive_controller_color(armlinkedtohaptic_left, arm_colors)
    right_color = _derive_controller_color(armlinkedtohaptic_right, arm_colors)

    parts = [
        f"arm left: {armlinkedtohaptic_left}",
        f"left instrument: {instrtype_to_name(instrtype_left)} ({color_to_name(left_color)})",
        f"arm right: {armlinkedtohaptic_right}",
        f"right instrument: {instrtype_to_name(instrtype_right)} ({color_to_name(right_color)})",
    ]
    return ". ".join(parts)
