import argparse
import glob
import os
import os.path as osp
import pickle
import sys

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = "/remote-home/linweihao"
sys.path.append(f"{PROJECT_ROOT}/structvla")

from train.dataset.normalize_pi0 import RunningStats, save


VIEWS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def sort_by_int(filename):
    return int(os.path.splitext(filename)[0])


def list_npy_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    return [osp.join(folder, f) for f in sorted(files, key=sort_by_int)]


def get_variant_dirs(dataset_path, base_prefixes):
    """
    Collect token dirs such as:
      maniparena_dualarm_codes_256_192
      maniparena_dualarm_recodes_256_192

    Priority:
      codes_* first, then recodes_*, and larger numeric suffix first.
    """
    dirs = []
    for base_prefix in base_prefixes:
        pattern = osp.join(dataset_path, f"{base_prefix}_*")
        dirs.extend([d for d in glob.glob(pattern) if osp.isdir(d)])

    dirs = list(dict.fromkeys(dirs))

    def variant_sort_key(path):
        name = osp.basename(path)
        if "_codes_" in name:
            group = 0
        elif "_recodes_" in name:
            group = 1
        else:
            group = 2
        return (group, name)

    return sorted(dirs, key=variant_sort_key)


def load_dualarm_episode(episode_dir, token_root, min_frames):
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


def main(dataset_path, output_path, normalizer_path, output_filename):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(normalizer_path, exist_ok=True)

    language_dir = osp.join(dataset_path, "real_all")
    if not osp.isdir(language_dir):
        raise ValueError(f"real_all not found under {dataset_path}")

    token_variant_dirs = get_variant_dirs(
        dataset_path,
        ["maniparena_dualarm_codes", "maniparena_dualarm_recodes"],
    )
    if not token_variant_dirs:
        raise ValueError(
            f"No maniparena_dualarm_codes_* / maniparena_dualarm_recodes_* dirs found under {dataset_path}"
        )

    print("Using dual-arm token dirs (priority order):")
    for d in token_variant_dirs:
        print("  ", d)

    min_frames = 8
    result_file = []

    episodes = [e for e in sorted(os.listdir(language_dir)) if osp.isdir(osp.join(language_dir, e))]
    print("Loading episodes from:", language_dir)

    for episode in tqdm(episodes):
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

        action = np.stack([np.load(p) for p in action_files], axis=0)
        if action.ndim == 1:
            action = action[:, None]
        if action.shape[-1] != 14:
            raise ValueError(
                f"Expected 14D dual-arm action for {episode}, got shape {action.shape}"
            )

        packed = False
        for token_root in token_variant_dirs:
            dualarm_tokens = load_dualarm_episode(episode_dir, token_root, min_frames)
            if dualarm_tokens is None:
                continue

            token_paths, token_len = dualarm_tokens
            usable_len = min(len(action), token_len)
            if usable_len < min_frames:
                continue

            result_file.append(
                {
                    "text": text,
                    "cam_high": token_paths["cam_high"][:usable_len],
                    "cam_left_wrist": token_paths["cam_left_wrist"][:usable_len],
                    "cam_right_wrist": token_paths["cam_right_wrist"][:usable_len],
                    "action": action[:usable_len].copy(),
                }
            )
            packed = True

        if not packed:
            continue

    print(f"Total packed samples (episode x variant): {len(result_file)}")
    if not result_file:
        raise ValueError("No valid samples found. Check your dataset path and token directories.")

    normalizer = RunningStats()
    action_data = np.concatenate([sample["action"] for sample in result_file], axis=0)
    normalizer.update(action_data)
    stats = normalizer.get_statistics()

    print("Mean:", stats.mean)
    print("Std:", stats.std)
    print("Q01:", stats.q01)
    print("Q99:", stats.q99)

    for sample in result_file:
        action = sample["action"]
        normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
        sample["action"] = np.clip(normalized, -1, 1)

    output_file = osp.join(output_path, output_filename)
    with open(output_file, "wb") as f:
        pickle.dump(result_file, f)
    print(f"Saved normalized data to {output_file}")

    save(normalizer_path, {"real_dualarm": stats})
    print(f"Saved normalizer statistics to {normalizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize ManipArena sim dataset action values.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/remote-home/linweihao/maniparena_dataset_real/processed_data",
        help="Root path to ManipArena sim processed_data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/remote-home/linweihao/maniparena_dataset_real/processed_data/meta",
        help="Path to save normalized data.",
    )
    parser.add_argument(
        "--normalizer_path",
        type=str,
        default="configs/normalizer_real",
        help="Path to save normalization stats.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="real_all_dualarm_norm.pkl",
        help="Filename for normalized pickle output.",
    )
    args = parser.parse_args()

    main(args.dataset_path, args.output_path, args.normalizer_path, args.output_filename)
