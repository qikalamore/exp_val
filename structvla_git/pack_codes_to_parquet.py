#!/usr/bin/env python3
"""Pack ManipArena code npy files into Parquet format."""

from __future__ import annotations

import argparse
import json
import os
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
    cameras: int = 0
    frames: int = 0
    parquet_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack /episode/camera/*.npy codes into one Parquet file in an output folder."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        default=Path("/remote-home/linweihao/maniparena_dataset_real/processed_data/maniparena_dualarm_codes_256_192"),
        help="Input directory, e.g. maniparena_dualarm_codes_256_192",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        default=Path("/remote-home/linweihao/maniparena_dataset_real/processed_data/maniparena_dualarm_codes_256_192_parquet_group"),
        help="Output directory to place parquet + metadata.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=".parquet",
        help=(
            "Output file suffix for each episode parquet (default: .parquet), "
            "e.g. _codes.parquet"
        ),
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="Parquet compression codec (default: zstd)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Rows buffered before each parquet write (default: 2000)",
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
        default=16,
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
            pa.field("camera", pa.string()),
            pa.field("frame_index", pa.int32()),
            pa.field("code_shape", pa.list_(pa.int32())),
            pa.field("code", pa.list_(pa.int64())),
        ]
    )


def make_empty_columns() -> dict[str, list[Any]]:
    return {
        "task": [],
        "episode": [],
        "episode_index": [],
        "camera": [],
        "frame_index": [],
        "code_shape": [],
        "code": [],
    }


def flush_batch(
    writer: pq.ParquetWriter,
    columns: dict[str, list[Any]],
    schema: pa.Schema,
) -> int:
    row_count = len(columns["task"])
    if row_count == 0:
        return 0

    table = pa.Table.from_pydict(columns, schema=schema)
    writer.write_table(table)

    for key in columns:
        columns[key].clear()

    return row_count


def pack_one_episode(
    episode_dir: Path,
    output_dir: Path,
    output_name: str,
    compression: str,
    batch_size: int,
) -> PackStats:
    schema = build_schema()
    codec = None if compression == "none" else compression

    columns = make_empty_columns()
    episode_parquet_path = output_dir / f"{episode_dir.name}{output_name}"

    stats = PackStats()

    writer = pq.ParquetWriter(
        where=str(episode_parquet_path),
        schema=schema,
        compression=codec,
        use_dictionary=False,
    )

    try:
        task_name, episode_index = parse_episode_info(episode_dir.name)
        camera_dirs = list_sorted_dirs(episode_dir)

        stats.episodes = 1
        stats.cameras = len(camera_dirs)

        for camera_dir in camera_dirs:
            frame_files = list_sorted_npy(camera_dir)
            camera_name = camera_dir.name

            for frame_file in frame_files:
                arr = np.load(frame_file, allow_pickle=False)
                flat = np.asarray(arr, dtype=np.int64).reshape(-1)

                frame_stem = frame_file.stem
                frame_index = int(frame_stem) if frame_stem.isdigit() else -1

                columns["task"].append(task_name)
                columns["episode"].append(episode_dir.name)
                columns["episode_index"].append(
                    episode_index if episode_index is not None else -1
                )
                columns["camera"].append(camera_name)
                columns["frame_index"].append(frame_index)
                columns["code_shape"].append([int(x) for x in arr.shape])
                columns["code"].append(flat.tolist())

                stats.frames += 1

                if len(columns["task"]) >= batch_size:
                    flush_batch(writer, columns, schema)

        flush_batch(writer, columns, schema)
        stats.parquet_files = 1
    finally:
        writer.close()

    return stats


def merge_stats(all_stats: list[PackStats]) -> PackStats:
    merged = PackStats()
    for s in all_stats:
        merged.episodes += s.episodes
        merged.cameras += s.cameras
        merged.frames += s.frames
        merged.parquet_files += s.parquet_files
    return merged


def pack_codes(
    input_dir: Path,
    output_dir: Path,
    output_name: str,
    compression: str,
    batch_size: int,
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
            "camera_dirs": 0,
            "frames": 0,
            "compression": compression,
            "batch_size": batch_size,
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
            s = pack_one_episode(
                episode_dir=episode_dir,
                output_dir=output_dir,
                output_name=output_name,
                compression=compression,
                batch_size=batch_size,
            )
            all_stats.append(s)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    pack_one_episode,
                    episode_dir,
                    output_dir,
                    output_name,
                    compression,
                    batch_size,
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
        "camera_dirs": stats.cameras,
        "frames": stats.frames,
        "compression": compression,
        "batch_size": batch_size,
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
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()