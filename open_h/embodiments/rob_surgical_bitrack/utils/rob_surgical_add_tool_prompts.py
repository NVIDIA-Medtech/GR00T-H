"""Add tool-augmented instruction text to Rob Surgical parquet files.

This script reads the existing instruction.text prompts and the per-frame tool
metadata columns (observation.meta.left_tool, observation.meta.right_tool,
observation.meta.aux_tool), then constructs a new instruction column that
prefixes the original prompt with structured tool information.

Output format per row:
    left tool: <name>. right tool: <name>. aux tool: <name>. <original prompt>

If a tool column is missing or empty, the literal string "none" is used.

The new column is written as `instruction.text_with_tool` into each episode
parquet file in-place.

Usage:
    uv run python open_h/embodiments/rob_surgical_bitrack/utils/rob_surgical_add_tool_prompts.py \
        --dataset-path /path/to/rob_surgical_dataset

    # Dry-run (print first 5 episodes, don't write):
    uv run python open_h/embodiments/rob_surgical_bitrack/utils/rob_surgical_add_tool_prompts.py \
        --dataset-path /path/to/rob_surgical_dataset --dry-run
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_PATH = "/path/to/rob_surgical_dataset"

# Parquet columns that contain the per-frame tool type string.
TOOL_COLUMNS = {
    "left": "observation.meta.left_tool",
    "right": "observation.meta.right_tool",
    "aux": "observation.meta.aux_tool",
}

# Source instruction column and the new output column.
SRC_INSTRUCTION_COL = "instruction.text"
DST_INSTRUCTION_COL = "instruction.text_with_tool"


def _safe_tool_name(value: object) -> str:
    """Return a cleaned tool name, falling back to 'none' for missing values.

    Args:
        value: Raw cell value from the tool metadata column.  May be a string,
               NaN, None, or empty string.

    Returns:
        Lowercase tool name string, or the literal string 'none'.

    Examples:
        >>> _safe_tool_name("bi_maryland_forceps")
        'bi_maryland_forceps'
        >>> _safe_tool_name(None)
        'none'
        >>> _safe_tool_name("")
        'none'
    """
    if value is None:
        return "none"
    s = str(value).strip().lower()
    if s in ("", "nan", "null"):
        return "none"
    return s


def build_tool_prefix(row: pd.Series) -> str:
    """Construct the tool-description prefix for a single row.

    Args:
        row: A pandas Series containing the tool metadata columns.

    Returns:
        A string in the format:
            'left tool: <name>. right tool: <name>. aux tool: <name>.'

    Examples:
        >>> import pandas as pd
        >>> row = pd.Series(
        ...     {
        ...         "observation.meta.left_tool": "mono_hook",
        ...         "observation.meta.right_tool": "forceps",
        ...         "observation.meta.aux_tool": "bi_maryland_forceps",
        ...     }
        ... )
        >>> build_tool_prefix(row)
        'left tool: mono_hook. right tool: forceps. aux tool: bi_maryland_forceps.'
    """
    parts = []
    for arm_label, col_name in TOOL_COLUMNS.items():
        tool_name = _safe_tool_name(row.get(col_name))
        parts.append(f"{arm_label} tool: {tool_name}")
    return ". ".join(parts) + "."


def build_augmented_instruction(row: pd.Series) -> str:
    """Build the full augmented instruction string for a single row.

    Combines the tool prefix with the original instruction text.  If the
    original instruction is missing or empty, only the tool prefix is returned.

    Args:
        row: A pandas Series with tool metadata and instruction.text columns.

    Returns:
        Augmented instruction string.

    Examples:
        >>> import pandas as pd
        >>> row = pd.Series(
        ...     {
        ...         "observation.meta.left_tool": "mono_hook",
        ...         "observation.meta.right_tool": "forceps",
        ...         "observation.meta.aux_tool": "bi_maryland_forceps",
        ...         "instruction.text": "The surgery process is a hemicolectomy.Suturing",
        ...     }
        ... )
        >>> build_augmented_instruction(row)
        'left tool: mono_hook. right tool: forceps. aux tool: bi_maryland_forceps. The surgery process is a hemicolectomy.Suturing'
    """
    prefix = build_tool_prefix(row)
    original = str(row.get(SRC_INSTRUCTION_COL, "")).strip()
    if not original or original.lower() in ("nan", "none"):
        return prefix
    return f"{prefix} {original}"


def process_episode(parquet_path: Path, dry_run: bool = False) -> int:
    """Add the instruction.text_with_tool column to a single episode parquet.

    Args:
        parquet_path: Path to the episode parquet file.
        dry_run: If True, print sample output but do not write.

    Returns:
        Number of rows processed.

    Raises:
        KeyError: If the source instruction column is missing.
    """
    df = pd.read_parquet(parquet_path)

    # Build augmented instruction for every row
    df[DST_INSTRUCTION_COL] = df.apply(build_augmented_instruction, axis=1)

    if dry_run:
        # Show first row as sample
        sample = df[DST_INSTRUCTION_COL].iloc[0]
        print(f'  {parquet_path.name}: "{sample}"')
    else:
        df.to_parquet(parquet_path, index=False)

    return len(df)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Add tool-augmented instruction text to Rob Surgical parquets."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(DEFAULT_DATASET_PATH),
        help="Path to the Rob Surgical LeRobot dataset root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample outputs for the first 5 episodes without writing.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: iterate all episode parquets and add augmented instructions.

    Raises:
        FileNotFoundError: If no parquet files are found at the dataset path.
    """
    args = parse_args()
    pattern = str(args.dataset_path / "data" / "chunk-*" / "episode_*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found matching: {pattern}")

    total_files = len(parquet_files)
    limit = 5 if args.dry_run else total_files
    mode_label = "DRY RUN" if args.dry_run else "WRITING"

    print(f"[{mode_label}] Processing {min(limit, total_files)} / {total_files} episodes...")
    print(f"  Source column: {SRC_INSTRUCTION_COL}")
    print(f"  Target column: {DST_INSTRUCTION_COL}")
    print()

    total_rows = 0
    for i, f in enumerate(parquet_files[:limit]):
        total_rows += process_episode(Path(f), dry_run=args.dry_run)

    print(f"\nDone. Processed {total_rows} total rows across {min(limit, total_files)} episodes.")


if __name__ == "__main__":
    main()
