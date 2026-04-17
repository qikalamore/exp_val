#!/usr/bin/env python3
"""Unpack ManipArena code parquet files back into /episode/camera/*.npy."""

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
    "camera",
    "frame_index",
    "code_shape",
    "code",
}


@dataclass
class UnpackStats:
    parquet_files: int = 0
    episodes: int = 0
    cameras: int = 0
    frames: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unpack parquet produced by pack_codes_to_parquet.py back into "
            "/episode/camera/*.npy."
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
        help="Directory to restore /episode/camera/*.npy under.",
    )
    parser.add_argument(
        "--index-width",
        type=int,
        default=0,
        help="Zero-pad width for numeric frame names (default: 0, disabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files if present.",
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


def format_frame_name(frame_index: int, row_index: int, index_width: int) -> str:
    if frame_index >= 0:
        if index_width > 0:
            return f"{frame_index:0{index_width}d}.npy"
        return f"{frame_index}.npy"
    if index_width > 0:
        return f"row_{row_index:0{index_width}d}.npy"
    return f"row_{row_index}.npy"


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
    if row_count == 0:
        return UnpackStats(parquet_files=1)

    seen_episodes: set[str] = set()
    seen_cameras: set[tuple[str, str]] = set()

    for row_idx in range(row_count):
        episode = str(data["episode"][row_idx])
        camera = str(data["camera"][row_idx])
        frame_index = int(data["frame_index"][row_idx])

        shape = tuple(int(x) for x in data["code_shape"][row_idx])
        flat = np.asarray(data["code"][row_idx], dtype=np.int64)
        arr = flat.reshape(shape)

        frame_name = format_frame_name(frame_index, row_idx, index_width)
        out_dir = output_dir / episode / camera
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / frame_name

        if out_file.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file without --overwrite: {out_file}"
            )

        np.save(out_file, arr, allow_pickle=False)

        seen_episodes.add(episode)
        seen_cameras.add((episode, camera))

    return UnpackStats(
        parquet_files=1,
        episodes=len(seen_episodes),
        cameras=len(seen_cameras),
        frames=row_count,
    )


def merge_stats(all_stats: list[UnpackStats]) -> UnpackStats:
    merged = UnpackStats()
    for stats in all_stats:
        merged.parquet_files += stats.parquet_files
        merged.episodes += stats.episodes
        merged.cameras += stats.cameras
        merged.frames += stats.frames
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
        file_iter = tqdm(parquet_files, desc="Unpacking code parquet", unit="file")

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
        "camera_dirs": stats.cameras,
        "frames": stats.frames,
        "index_width": args.index_width,
        "overwrite": args.overwrite,
    }

    meta_path = output_dir / "unpack_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
