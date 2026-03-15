"""Add per-timestep state-derived language prompts to CMR parquet files.

This script reads the per-frame observation.state column from each episode parquet,
extracts instrument type, arm color, and arm-to-haptic linkage, and constructs a
natural-language prompt that encodes this information. The prompt is written as a new
column `instruction.text_with_state` directly into each parquet file in-place.

This replaces the previous approach of sending these values as always-visible state
inputs to the DiT. Instead, the information goes through the VLM backbone as language.

Output format per row:
    arm left: <linked_val>. left instrument: <instr_name> (<color_name>).
    arm right: <linked_val>. right instrument: <instr_name> (<color_name>).
    do a <procedure>

Observation.state index mapping:
    [2:6]  -> arm_0_color through arm_3_color (color enum per arm slot)
    [20]   -> armlinkedtohaptic_left  (which arm slot the left controller uses, -1 = none)
    [21]   -> armlinkedtohaptic_right (which arm slot the right controller uses, -1 = none)
    [22]   -> instrtype_left  (instrument type enum for left controller)
    [23]   -> instrtype_right (instrument type enum for right controller)

Color derivation:
    The color for each controller is derived from armlinkedtohaptic:
    - If linked >= 0: color = arm_{linked}_color (observation.state[2 + linked])
    - If linked == -1: color = 0 (None / not connected)

Usage:
    # Dry-run (print first 5 episodes per dataset, don't write):
    python open_h/embodiments/cmr_versius/utils/cmr_add_state_prompts.py --dry-run

    # Write in-place:
    python open_h/embodiments/cmr_versius/utils/cmr_add_state_prompts.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path

import pandas as pd


try:
    from cmr_state_prompt_prefix import build_state_prefix_from_fields, extract_state_prompt_fields
except ModuleNotFoundError:
    # Fallback for environments importing this script via repository-root paths.
    from open_h.embodiments.cmr_versius.utils.cmr_state_prompt_prefix import (
        build_state_prefix_from_fields,
        extract_state_prompt_fields,
    )

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Root directory containing the 4 CMR datasets
CMR_ROOT = Path(
    os.environ.get("OPEN_H_DATA_PATH", "."),
    "cmr-surgical",
)

# Only process the 4 variants
TARGET_DATASETS = [
    "cholecystectomy",
    "hysterectomy",
    "inguinal_hernia",
    "prostatectomy",
]

# Folder name -> procedure string for the language prompt suffix
PROCEDURE_MAP = {
    "cholecystectomy": "do a cholecystectomy",
    "hysterectomy": "do a hysterectomy",
    "inguinal_hernia": "do an inguinal hernia repair",
    "prostatectomy": "do a prostatectomy",
}

# Destination column written into each parquet
DST_COL = "instruction.text_with_state"

# ──────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────


def build_prompt(state: Sequence[float], procedure: str) -> str:
    """Build the full per-timestep language prompt from observation.state.

    Format:
        arm left: <linked>. left instrument: <name> (<color>).
        arm right: <linked>. right instrument: <name> (<color>).
        do a <procedure>

    Args:
        state: The full observation.state array for this timestep (list or np array).
        procedure: The procedure suffix string (e.g. "do a prostatectomy").

    Returns:
        Complete prompt string for this timestep.
    """
    prefix_fields = extract_state_prompt_fields(state)
    prefix = build_state_prefix_from_fields(**prefix_fields)
    return f"{prefix}. {procedure}"


# ──────────────────────────────────────────────────────────────────────
# Episode processing
# ──────────────────────────────────────────────────────────────────────


def process_episode(parquet_path: Path, procedure: str, dry_run: bool = False) -> int:
    """Add the instruction.text_with_state column to a single episode parquet.

    Reads the observation.state column, constructs a per-row prompt string, and
    writes it back as a new column. The original parquet is overwritten in-place.

    Args:
        parquet_path: Path to the episode parquet file.
        procedure: Procedure suffix string (e.g. "do a prostatectomy").
        dry_run: If True, print sample output but do not write.

    Returns:
        Number of rows processed.
    """
    df = pd.read_parquet(parquet_path)

    # observation.state is stored as a list-of-floats per row
    state_col = df["observation.state"]

    # Vectorized prompt construction: apply build_prompt to each row's state
    prompts = state_col.apply(lambda s: build_prompt(s, procedure))
    df[DST_COL] = prompts

    if dry_run:
        # Show first and last row as samples
        print(f"  {parquet_path.name} ({len(df)} rows):")
        print(f'    row 0: "{prompts.iloc[0]}"')
        if len(df) > 1:
            print(f'    row {len(df) - 1}: "{prompts.iloc[-1]}"')
    else:
        df.to_parquet(parquet_path, index=False)

    return len(df)


def get_episode_parquets(dataset_root: Path) -> list[Path]:
    """Discover all episode parquet files for a dataset using info.json metadata.

    Reads info.json to get the data_path pattern, total_episodes, and chunks_size,
    then constructs the path to each episode parquet.

    Args:
        dataset_root: Root directory of the dataset (e.g. .../cholecystectomy).

    Returns:
        Sorted list of Path objects to episode parquet files.

    Raises:
        FileNotFoundError: If info.json is missing.
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    info = json.loads(info_path.read_text())
    data_path_pattern = info["data_path"]
    total_episodes = info["total_episodes"]
    chunks_size = info["chunks_size"]

    parquets = []
    for ep_idx in range(total_episodes):
        chunk = ep_idx // chunks_size
        rel_path = data_path_pattern.format(episode_chunk=chunk, episode_index=ep_idx)
        parquets.append(dataset_root / rel_path)

    return parquets


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Add per-timestep state-derived language prompts to CMR parquets."
    )
    parser.add_argument(
        "--cmr-root",
        type=Path,
        default=CMR_ROOT,
        help="Root directory containing the CMR dataset folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample outputs for the first 5 episodes per dataset without writing.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: iterate all 4 CMR datasets and add state-derived prompts.

    For each dataset, discovers episode parquets via info.json, constructs per-row
    prompts from observation.state fields, and writes them as a new parquet column.

    Raises:
        FileNotFoundError: If dataset directories or info.json files are missing.
        SystemExit: If no target datasets are found.
    """
    args = parse_args()
    mode_label = "DRY RUN" if args.dry_run else "WRITING"

    print(f"[{mode_label}] CMR state-to-language prompt writer")
    print(f"  Root: {args.cmr_root}")
    print(f"  Target column: {DST_COL}")
    print()

    grand_total_rows = 0
    grand_total_episodes = 0

    for ds_name in TARGET_DATASETS:
        ds_root = args.cmr_root / ds_name
        if not ds_root.exists():
            print(f"WARNING: Dataset directory not found, skipping: {ds_root}")
            continue

        procedure = PROCEDURE_MAP[ds_name]
        parquets = get_episode_parquets(ds_root)
        total_eps = len(parquets)
        limit = 5 if args.dry_run else total_eps

        print(f'--- {ds_name} ({total_eps} episodes, procedure="{procedure}") ---')

        ds_rows = 0
        for i, pq_path in enumerate(parquets[:limit]):
            if not pq_path.exists():
                print(f"  WARNING: Missing parquet: {pq_path}")
                continue
            ds_rows += process_episode(pq_path, procedure, dry_run=args.dry_run)

            # Progress reporting every 500 episodes (when not dry-run)
            if not args.dry_run and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{total_eps} episodes ({ds_rows} rows so far)...")

        print(f"  => {min(limit, total_eps)} episodes, {ds_rows} rows")
        print()
        grand_total_rows += ds_rows
        grand_total_episodes += min(limit, total_eps)

    print(f"Done. Processed {grand_total_rows} total rows across {grand_total_episodes} episodes.")


if __name__ == "__main__":
    main()
