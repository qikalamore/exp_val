#!/usr/bin/env python3
"""Pack ManipArena real_all episodes into Parquet with actions and instruction only."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class PackStats:
    episodes: int = 0
    actions: int = 0
    parquet_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack /episode/actions/*.npy and instruction.txt into one Parquet file per episode."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("/remote-home/linweihao/maniparena_dataset_real/processed_data/real_all"),
        help="Input directory containing episode folders.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path(
            "/remote-home/linweihao/maniparena_dataset_real/processed_data/real_all_actions_instruction_parquet_group"
        ),
        help="Output directory to place parquet files and metadata.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=".parquet",
        help="Output file suffix for each episode parquet (default: .parquet).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="Parquet compression codec (default: zstd).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional debug limit for number of episodes.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of worker processes for episode-level parallel packing.",
    )
    return parser.parse_args()


def parse_episode_info(episode_name: str) -> tuple[str, int | None]:
    task_name = episode_name
    episode_index: int | None = None

    if "_chunk-" in episode_name:
        task_name = episode_name.split("_chunk-", maxsplit=1)[0]

    marker = "_episode_"
    if marker in episode_name:
        tail = episode_name.rsplit(marker, maxsplit=1)[-1]
        if tail.isdigit():
            episode_index = int(tail)

    return task_name, episode_index


def list_sorted_dirs(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir()])


def list_sorted_npy(path: Path) -> list[Path]:
    def key_fn(file_path: Path) -> tuple[int, str]:
        stem = file_path.stem
        return (int(stem), stem) if stem.isdigit() else (10**9, stem)

    return sorted(path.glob("*.npy"), key=key_fn)


def build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("task", pa.string()),
            pa.field("episode", pa.string()),
            pa.field("episode_index", pa.int32()),
            pa.field("instruction", pa.string()),
            pa.field("num_actions", pa.int32()),
            pa.field("action_shape", pa.list_(pa.int32())),
            pa.field("actions", pa.list_(pa.list_(pa.float32()))),
        ]
    )


def read_instruction(episode_dir: Path) -> str:
    instruction_path = episode_dir / "instruction.txt"
    if not instruction_path.is_file():
        raise FileNotFoundError(f"Missing instruction.txt in {episode_dir}")
    return instruction_path.read_text(encoding="utf-8").strip()


def pack_one_episode(
    episode_dir: Path,
    output_dir: Path,
    output_name: str,
    compression: str,
) -> PackStats:
    schema = build_schema()
    codec = None if compression == "none" else compression
    episode_parquet_path = output_dir / f"{episode_dir.name}{output_name}"

    task_name, episode_index = parse_episode_info(episode_dir.name)
    instruction = read_instruction(episode_dir)
    action_dir = episode_dir / "actions"
    action_files = list_sorted_npy(action_dir)

    actions: list[list[float]] = []
    action_shape: list[int] | None = None

    for action_file in action_files:
        arr = np.load(action_file, allow_pickle=False)
        if action_shape is None:
            action_shape = [int(x) for x in arr.shape]
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        actions.append(flat.tolist())

    if action_shape is None:
        action_shape = [0]

    columns: dict[str, list[Any]] = {
        "task": [task_name],
        "episode": [episode_dir.name],
        "episode_index": [episode_index if episode_index is not None else -1],
        "instruction": [instruction],
        "num_actions": [len(actions)],
        "action_shape": [action_shape],
        "actions": [actions],
    }

    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(
        table,
        where=str(episode_parquet_path),
        compression=codec,
        use_dictionary=False,
    )

    return PackStats(episodes=1, actions=len(actions), parquet_files=1)


def merge_stats(all_stats: list[PackStats]) -> PackStats:
    merged = PackStats()
    for s in all_stats:
        merged.episodes += s.episodes
        merged.actions += s.actions
        merged.parquet_files += s.parquet_files
    return merged


def pack_codes(
    input_dir: Path,
    output_dir: Path,
    output_name: str,
    compression: str,
    max_episodes: int | None,
    num_workers: int,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = list_sorted_dirs(input_dir)
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    if not episode_dirs:
        meta = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "episodes": 0,
            "parquet_files": 0,
            "actions": 0,
            "compression": compression,
            "file_suffix": output_name,
            "num_workers": num_workers,
        }
        meta_path = output_dir / "pack_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return

    all_stats: list[PackStats] = []

    if num_workers <= 1:
        episode_iter = episode_dirs
        if tqdm is not None:
            episode_iter = tqdm(episode_dirs, desc="Packing episodes", unit="episode")

        for episode_dir in episode_iter:
            all_stats.append(
                pack_one_episode(
                    episode_dir=episode_dir,
                    output_dir=output_dir,
                    output_name=output_name,
                    compression=compression,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    pack_one_episode,
                    episode_dir,
                    output_dir,
                    output_name,
                    compression,
                )
                for episode_dir in episode_dirs
            ]

            future_iter = as_completed(futures)
            if tqdm is not None:
                future_iter = tqdm(
                    future_iter,
                    total=len(futures),
                    desc="Packing episodes",
                    unit="episode",
                )

            for future in future_iter:
                all_stats.append(future.result())

    stats = merge_stats(all_stats)

    meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "episodes": stats.episodes,
        "parquet_files": stats.parquet_files,
        "actions": stats.actions,
        "compression": compression,
        "file_suffix": output_name,
        "num_workers": num_workers,
    }

    meta_path = output_dir / "pack_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    pack_codes(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        output_name=args.output_name,
        compression=args.compression,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
