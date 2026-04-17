#!/usr/bin/env python3
"""Unpack ManipArena action/instruction parquet back into /episode/actions/*.npy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REQUIRED_COLUMNS = {
    "task",
    "episode",
    "episode_index",
    "instruction",
    "num_actions",
    "action_shape",
    "actions",
}


@dataclass
class UnpackStats:
    parquet_files: int = 0
    episodes: int = 0
    actions: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unpack parquet produced by "
            "pack_real_all_actions_instruction_to_parquet.py back into "
            "/episode/instruction.txt + /episode/actions/*.npy."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A parquet file or a directory containing parquet files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to restore /episode/instruction.txt + /actions/*.npy under.",
    )
    parser.add_argument(
        "--index-width",
        type=int,
        default=0,
        help="Zero-pad width for numeric action names (default: 0, disabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing instruction.txt / .npy files if present.",
    )
    return parser.parse_args()


def collect_parquet_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".parquet":
            raise ValueError(f"Input file is not a parquet file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return sorted(p for p in input_path.glob("*.parquet") if p.is_file())


def validate_columns(columns: list[str], parquet_path: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(
            f"Parquet missing required columns {sorted(missing)}: {parquet_path}"
        )


def unpack_one_file(
    parquet_path: Path,
    output_dir: Path,
    index_width: int,
    overwrite: bool,
) -> UnpackStats:
    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    columns = list(data.keys())
    validate_columns(columns, parquet_path)

    row_count = len(data["episode"])
    total_actions = 0

    for row_idx in range(row_count):
        episode = str(data["episode"][row_idx])
        instruction = str(data["instruction"][row_idx])
        action_shape = tuple(int(x) for x in data["action_shape"][row_idx])
        actions = data["actions"][row_idx]

        episode_dir = output_dir / episode
        actions_dir = episode_dir / "actions"
        episode_dir.mkdir(parents=True, exist_ok=True)
        actions_dir.mkdir(parents=True, exist_ok=True)

        instruction_path = episode_dir / "instruction.txt"
        if instruction_path.exists() and not overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing instruction.txt without "
                f"--overwrite: {instruction_path}"
            )
        instruction_path.write_text(instruction, encoding="utf-8")

        for action_idx, flat_action in enumerate(actions):
            arr = np.asarray(flat_action, dtype=np.float32)
            if action_shape != (0,):
                arr = arr.reshape(action_shape)

            if index_width > 0:
                action_name = f"{action_idx:0{index_width}d}.npy"
            else:
                action_name = f"{action_idx}.npy"
            action_path = actions_dir / action_name
            if action_path.exists() and not overwrite:
                raise FileExistsError(
                    "Refusing to overwrite existing action file without "
                    f"--overwrite: {action_path}"
                )

            np.save(action_path, arr, allow_pickle=False)
            total_actions += 1

    return UnpackStats(
        parquet_files=1,
        episodes=row_count,
        actions=total_actions,
    )


def merge_stats(all_stats: list[UnpackStats]) -> UnpackStats:
    merged = UnpackStats()
    for stats in all_stats:
        merged.parquet_files += stats.parquet_files
        merged.episodes += stats.episodes
        merged.actions += stats.actions
    return merged


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = collect_parquet_files(input_path)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {input_path}")

    file_iter: list[Path] | Any = parquet_files
    if tqdm is not None and len(parquet_files) > 1:
        file_iter = tqdm(
            parquet_files,
            desc="Unpacking action parquet",
            unit="file",
        )

    all_stats: list[UnpackStats] = []
    for parquet_path in file_iter:
        all_stats.append(
            unpack_one_file(
                parquet_path=parquet_path,
                output_dir=output_dir,
                index_width=args.index_width,
                overwrite=args.overwrite,
            )
        )

    stats = merge_stats(all_stats)
    meta = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "parquet_files": stats.parquet_files,
        "episodes": stats.episodes,
        "actions": stats.actions,
        "index_width": args.index_width,
        "overwrite": args.overwrite,
    }

    meta_path = output_dir / "unpack_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
