import argparse
import glob
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from train.dataset.normalize_pi0 import RunningStats, save


VIEWS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def sort_by_int(filename: str) -> int:
    return int(os.path.splitext(filename)[0])


def list_npy_files(folder: str) -> list[str]:
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    return [osp.join(folder, f) for f in sorted(files, key=sort_by_int)]


def normalize_action(action: np.ndarray) -> np.ndarray:
    normalized = action.copy()
    for i in range(action.shape[-1]):
        if i not in [6, 13]:
            normalized[:, i] = action[:, i] - action[0, i]
    return normalized


def normalize_action_pose(action: np.ndarray) -> np.ndarray:
    normalized = action.copy()
    state = action[0].copy()
    mask = np.ones_like(state, dtype=bool)
    mask[6] = False
    mask[13] = False
    dims = mask.shape[0]
    delta = np.where(mask, state, 0)
    normalized[..., :dims] -= np.expand_dims(delta, axis=-2)
    normalized[..., 3:6] -= 2 * np.pi * np.round(normalized[..., 3:6] / (2 * np.pi))
    normalized[..., 10:13] -= 2 * np.pi * np.round(normalized[..., 10:13] / (2 * np.pi))
    return normalized


def get_variant_dirs(dataset_path: str, base_prefixes: list[str]) -> list[str]:
    dirs = []
    for base_prefix in base_prefixes:
        pattern = osp.join(dataset_path, f"{base_prefix}_*")
        dirs.extend([d for d in glob.glob(pattern) if osp.isdir(d)])

    dirs = list(dict.fromkeys(dirs))

    def variant_sort_key(path: str) -> tuple[int, str]:
        name = osp.basename(path)
        if "_codes_" in name:
            group = 0
        elif "_recodes_" in name:
            group = 1
        else:
            group = 2
        return (group, name)

    return sorted(dirs, key=variant_sort_key)


def load_episode_tokens(episode_dir: str, token_root: str, min_frames: int) -> tuple[dict[str, list[str]], int] | None:
    episode_tokens_root = osp.join(token_root, osp.basename(episode_dir))
    if not osp.isdir(episode_tokens_root):
        return None

    token_paths = {}
    lengths = []
    for view in VIEWS:
        view_dir = osp.join(episode_tokens_root, view)
        if not osp.isdir(view_dir):
            return None
        files = list_npy_files(view_dir)
        if len(files) < min_frames:
            return None
        token_paths[view] = files
        lengths.append(len(files))

    return token_paths, min(lengths)


def infer_task_name(episode_name: str) -> str:
    if "_chunk-" in episode_name:
        return episode_name.split("_chunk-", 1)[0]
    return episode_name


def build_samples(
    action: np.ndarray,
    text: str,
    token_paths: dict[str, list[str]],
    usable_len: int,
    use_pose: bool,
    use_chunk: bool,
    chunk_size: int,
    chunk_stride: int,
) -> list[dict]:
    samples = []
    action = action[:usable_len].copy()

    if use_chunk:
        if usable_len < chunk_size:
            return samples
        for start in range(0, usable_len - chunk_size + 1, chunk_stride):
            end = start + chunk_size
            chunk_action = action[start:end].copy()
            samples.append(
                {
                    "text": text,
                    "cam_high": token_paths["cam_high"][start:end],
                    "cam_left_wrist": token_paths["cam_left_wrist"][start:end],
                    "cam_right_wrist": token_paths["cam_right_wrist"][start:end],
                    "action": normalize_action_pose(chunk_action) if use_pose else normalize_action(chunk_action),
                }
            )
        return samples

    samples.append(
        {
            "text": text,
            "cam_high": token_paths["cam_high"][:usable_len],
            "cam_left_wrist": token_paths["cam_left_wrist"][:usable_len],
            "cam_right_wrist": token_paths["cam_right_wrist"][:usable_len],
            "action": normalize_action_pose(action) if use_pose else normalize_action(action),
        }
    )
    return samples


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.normalizer_path, exist_ok=True)

    language_dir = osp.join(args.dataset_path, "sim_all")
    if not osp.isdir(language_dir):
        raise ValueError(f"sim_all not found under {args.dataset_path}")

    token_variant_dirs = get_variant_dirs(
        args.dataset_path,
        ["maniparena_dualarm_codes", "maniparena_dualarm_recodes"],
    )
    if not token_variant_dirs:
        raise ValueError(
            "No maniparena_dualarm_codes_* or maniparena_dualarm_recodes_* dirs found "
            f"under {args.dataset_path}"
        )

    min_frames = args.chunk_size if args.use_chunk else args.min_frames
    task_samples = defaultdict(list)

    print("Using token dirs:")
    for token_dir in token_variant_dirs:
        print("  ", token_dir)

    episodes = [e for e in sorted(os.listdir(language_dir)) if osp.isdir(osp.join(language_dir, e))]
    print("Loading episodes from:", language_dir)

    for episode in tqdm(episodes, desc="Processing ManipArena episodes"):
        episode_dir = osp.join(language_dir, episode)
        instr_file = osp.join(episode_dir, "instruction.txt")
        action_dir = osp.join(episode_dir, "actions")

        if not osp.isfile(instr_file) or not osp.isdir(action_dir):
            continue

        with open(instr_file, "r") as f:
            text = f.read().strip()

        action_files = list_npy_files(action_dir)
        if len(action_files) < min_frames:
            continue

        action = np.stack([np.load(path) for path in action_files], axis=0).astype(np.float32)
        if action.ndim == 1:
            action = action[:, None]
        if action.shape[-1] != 14:
            raise ValueError(
                f"Expected 14D dual-arm action for {episode}, got shape {action.shape}"
            )

        task_name = infer_task_name(episode)

        for token_root in token_variant_dirs:
            loaded = load_episode_tokens(episode_dir, token_root, min_frames)
            if loaded is None:
                continue

            token_paths, token_len = loaded
            usable_len = min(len(action), token_len)
            if usable_len < min_frames:
                continue

            samples = build_samples(
                action=action,
                text=text,
                token_paths=token_paths,
                usable_len=usable_len,
                use_pose=args.use_pose,
                use_chunk=args.use_chunk,
                chunk_size=args.chunk_size,
                chunk_stride=args.chunk_stride,
            )
            task_samples[task_name].extend(samples)

    total_samples = sum(len(samples) for samples in task_samples.values())
    print(f"Collected {total_samples} samples across {len(task_samples)} tasks")
    if total_samples == 0:
        raise ValueError("No valid samples found. Check dataset_path and token directories.")

    result_file = []
    norm_stats_save = {}

    for task_name, samples in task_samples.items():
        action_data = np.concatenate([sample["action"] for sample in samples], axis=0)
        normalizer = RunningStats()
        normalizer.update(action_data)
        stats = normalizer.get_statistics()
        norm_stats_save[task_name] = stats

        for sample in samples:
            action = sample["action"].copy()
            normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
            sample["action"] = np.clip(normalized, -1, 1)

        result_file.extend(samples)
        print(f"Task {task_name}: {len(samples)} samples")

    output_file = osp.join(args.output_path, args.output_filename)
    with open(output_file, "wb") as f:
        pickle.dump(result_file, f)

    save(args.normalizer_path, norm_stats_save)

    print(f"Saved dataset to: {output_file}")
    print(f"Saved task-wise normalizer stats to: {args.normalizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ManipArena sim pickle with ALOHA-style pose delta + task-wise sepnorm."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/remote-home/linweihao/maniparena_dataset_sim/processed_data",
        help="Root path to ManipArena processed_data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/remote-home/linweihao/maniparena_dataset_sim/processed_data/aloha_meta",
        help="Directory to save the normalized pickle.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="sim_all_dualarm_sepnorm.pkl",
        help="Filename for the generated pickle.",
    )
    parser.add_argument(
        "--normalizer_path",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "normalizer_sim_sepnorm_aloha"),
        help="Directory to save task-wise norm_stats.json.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=8,
        help="Minimum usable frames per episode when not chunking.",
    )
    parser.add_argument(
        "--use_pose",
        action="store_true",
        default=True,
        help="Apply pose-delta normalization before q01/q99 scaling.",
    )
    parser.add_argument(
        "--no_use_pose",
        dest="use_pose",
        action="store_false",
        help="Disable pose-delta normalization and use plain delta normalization.",
    )
    parser.add_argument(
        "--use_chunk",
        action="store_true",
        help="Slice each episode into fixed-length chunks before normalization.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=20,
        help="Chunk size when --use_chunk is enabled.",
    )
    parser.add_argument(
        "--chunk_stride",
        type=int,
        default=1,
        help="Sliding-window stride when --use_chunk is enabled.",
    )
    main(parser.parse_args())
