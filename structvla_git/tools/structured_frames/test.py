#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-arm simulator keystep extractor with real-robot-style defaults.

This script reuses the dual-arm data handling and manifest generation logic from
structured_frames_extract_sim.py, while defaulting the keyframe thresholds to
the same values used by extract_real_robot.py.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import structured_frames_extract_sim as sim_utils


REAL_STYLE_DEFAULTS = {
    "min_changes": 0,
    "max_lookahead": 15,
    "min_offset_after_flip": 0,
    "pre_move_horizon": 3,
    "grip_settle_eps": 0.05,
    "grip_settle_win": 3,
    "min_window": 2,
    "mag_thresh": 0.2,
    "flip_pos_pctl": 65.0,
    "flip_rot_pctl": 65.0,
    "flip_scale": 1.0,
    "flip_pos_min": 0.35,
    "flip_rot_min": 0.35,
    "flip_ema_win": 3,
    "flip_hysteresis": 0.8,
    "still_pos_pctl": 20.0,
    "still_rot_pctl": 25.0,
    "still_scale": 1.5,
    "still_pos_max": 0.35,
    "still_rot_max": 0.40,
    "still_win": 4,
    "still_min_gap": 15,
    "still_flip_guard": 2,
    "still_max_scale": 2.0,
    "still_ema_win": 3,
    "still_hysteresis": 1.0,
    "still_ratio": 0.6,
    "still_cp_guard": 0,
    "still_start_offset": 0,
    "still_relax_dist": 0,
    "still_relax_scale": 1.0,
    "still_relax_ratio": 0.0,
    "bad_cp_min_interval": 0,
    "skip_bad_cp_episode": False,
    "max_cp_per_episode": 0,
    "backfill_min_gap": 10,
    "backfill_tail_offset": 10,
    "no_still_keys": False,
}


def build_parser() -> argparse.ArgumentParser:
    d = REAL_STYLE_DEFAULTS
    ap = argparse.ArgumentParser(description="Dual-arm simulator extractor with real-robot defaults.")
    ap.add_argument("--dataset", type=str,default="/remote-home/linweihao/maniparena_dataset_real/processed_data/meta/real_all_dualarm_norm.pkl", required=False)
    ap.add_argument("--out_dir", type=str,default="/remote-home/linweihao/maniparena_dataset_real/processed_data/structout_testset", required=False)
    ap.add_argument("--test", action="store_true")
    for name in [
        "min_changes", "max_lookahead", "min_offset_after_flip", "pre_move_horizon",
        "grip_settle_win", "min_window", "flip_ema_win", "still_win", "still_min_gap",
        "still_flip_guard", "still_ema_win", "backfill_min_gap", "backfill_tail_offset",
        "still_cp_guard", "still_start_offset", "still_relax_dist",
        "bad_cp_min_interval", "max_cp_per_episode",
    ]:
        ap.add_argument(f"--{name}", type=int, default=d[name])
    for name in [
        "grip_settle_eps", "mag_thresh", "flip_pos_pctl", "flip_rot_pctl", "flip_scale",
        "flip_pos_min", "flip_rot_min", "flip_hysteresis", "still_pos_pctl",
        "still_rot_pctl", "still_scale", "still_pos_max", "still_rot_max",
        "still_max_scale", "still_hysteresis", "still_ratio", "still_relax_scale",
        "still_relax_ratio",
    ]:
        ap.add_argument(f"--{name}", type=float, default=d[name])
    ap.add_argument("--no_still_keys", action="store_true", default=d["no_still_keys"])
    ap.add_argument("--skip_bad_cp_episode", action="store_true", default=d["skip_bad_cp_episode"])
    return ap


def run(args: argparse.Namespace) -> None:
    frames = sim_utils.flatten_pkl_libero(args.dataset, test=args.test)
    grouped: Dict[str, List[int]] = {}
    for i, fr in enumerate(frames):
        grouped.setdefault(fr["episode_id"], []).append(i)

    sim_utils.log(f"Episodes detected: {len(grouped)}")
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []

    for ep_id, indices in tqdm(grouped.items(), desc="Episodes"):
        steps = [int(frames[i]["step"]) for i in indices]
        actions = np.stack([np.asarray(frames[i]["action"], dtype=np.float32) for i in indices], axis=0)
        ep_name = frames[indices[0]].get("episode_name", str(ep_id))
        first_idx_global = int(indices[0])
        first_step = int(frames[first_idx_global]["step"])
        action_first = list(map(float, np.asarray(frames[first_idx_global]["action"]).tolist()))
        all_candidates: List[Dict[str, Any]] = []

        for arm_name, arm_actions, g_values in sim_utils.get_arm_action_views(actions):
            mode_used = sim_utils.SIM_GRIPPER_BINARIZE_MODE
            grip_bits = sim_utils.binarize_gripper_sequence(g_values).tolist()
            cps = sim_utils.find_change_points(grip_bits) if len(grip_bits) >= 2 else []
            if len(cps) < args.min_changes:
                cps = []

            th = sim_utils.compute_episode_thresholds(
                arm_actions,
                args.flip_pos_pctl, args.flip_rot_pctl,
                args.still_pos_pctl, args.still_rot_pctl,
                args.flip_scale, args.still_scale,
                args.flip_pos_min, args.flip_rot_min,
                args.still_pos_max, args.still_rot_max,
            )

            arm_keysteps: List[Dict[str, Any]] = []
            for cp in cps:
                before_rel = cp - 1
                if before_rel < 0:
                    continue
                ks_idx_global, lookahead_used, pos_at_ks, rot_at_ks, found_premove = sim_utils.pick_keystep_after_grip_settle_libero(
                    frames, indices, cp, grip_bits, g_values,
                    th["flip_enter_pos"], th["flip_enter_rot"],
                    args.flip_hysteresis, args.flip_ema_win,
                    args.max_lookahead, args.min_offset_after_flip,
                    args.grip_settle_eps, args.grip_settle_win,
                    args.pre_move_horizon, actions_view=arm_actions,
                )
                step_before = int(frames[int(indices[before_rel])]["step"])
                step_keystep = int(frames[ks_idx_global]["step"])
                window_len = max(1, step_keystep - step_before + 1)
                if window_len <= max(args.min_window, 1):
                    ks_rel = steps.index(step_keystep)
                    win_actions = arm_actions[before_rel: ks_rel + 1, :]
                    if sim_utils.window_action_magnitude(win_actions) < args.mag_thresh:
                        continue
                arm_keysteps.append({
                    "type": "flip",
                    "cp": int(cp),
                    "idx_before": int(indices[before_rel]),
                    "idx_keystep": int(ks_idx_global),
                    "step_before": int(step_before),
                    "step_keystep": int(step_keystep),
                    "gripper_before": int(grip_bits[cp - 1]),
                    "gripper_after": int(grip_bits[cp]),
                    "pos_delta_keystep": float(pos_at_ks),
                    "rot_delta_keystep": float(rot_at_ks),
                    "lookahead_used": int(lookahead_used),
                    "binarize_mode": mode_used,
                    "found_premove": bool(found_premove),
                    "event_arm": arm_name,
                    "flip_enter_pos": float(th["flip_enter_pos"]),
                    "flip_enter_rot": float(th["flip_enter_rot"]),
                    "still_pos_thr": float(th["still_pos"]),
                    "still_rot_thr": float(th["still_rot"]),
                })

            if not args.no_still_keys:
                stills = sim_utils.collect_still_keysteps_libero(
                    frames=frames,
                    indices=indices,
                    steps=steps,
                    actions=arm_actions,
                    grip_bits=grip_bits,
                    mode_used=mode_used,
                    pos_mag_thresh=th["still_pos"],
                    rot_mag_thresh=th["still_rot"],
                    win=max(1, int(args.still_win)),
                    min_gap=max(1, int(args.still_min_gap)),
                    existing_steps=[k["step_keystep"] for k in arm_keysteps],
                    flip_steps=[k["step_keystep"] for k in arm_keysteps],
                    flip_guard=int(args.still_flip_guard),
                    max_scale=float(args.still_max_scale),
                    ema_win=int(args.still_ema_win),
                    hysteresis=float(args.still_hysteresis),
                    ratio_req=float(args.still_ratio),
                    start_step=int(steps[0]) + int(args.still_start_offset),
                    cp_steps=[steps[cp] for cp in cps] if cps else [],
                    cp_guard=int(args.still_cp_guard),
                    relax_dist=int(args.still_relax_dist) if args.still_relax_dist > 0 else None,
                    relax_scale=float(args.still_relax_scale),
                    relax_ratio=float(args.still_relax_ratio),
                )
                for item in stills:
                    item["flip_enter_pos"] = float(th["flip_enter_pos"])
                    item["flip_enter_rot"] = float(th["flip_enter_rot"])
                    item["still_pos_thr"] = float(th["still_pos"])
                    item["still_rot_thr"] = float(th["still_rot"])
                arm_keysteps.extend(sim_utils.attach_event_arm(stills, arm_name))

            selected_steps = [int(k["step_keystep"]) for k in arm_keysteps]
            min_gap = int(args.backfill_min_gap)
            first_step_in_ep = int(steps[0])
            last_step_in_ep = int(steps[-1])

            def append_if_valid(prefer_step: int, forbid: List[int], collected: List[Dict[str, Any]]) -> None:
                cand = sim_utils.nearest_valid_step(prefer_step, steps, forbid, min_gap)
                if cand is None:
                    return
                item = sim_utils.make_backfill_item(frames, indices, steps, arm_actions, grip_bits, mode_used, cand)
                item["event_arm"] = arm_name
                item["flip_enter_pos"] = float(th["flip_enter_pos"])
                item["flip_enter_rot"] = float(th["flip_enter_rot"])
                item["still_pos_thr"] = float(th["still_pos"])
                item["still_rot_thr"] = float(th["still_rot"])
                collected.append(item)
                forbid.append(int(cand))

            backfills: List[Dict[str, Any]] = []
            if len(selected_steps) == 0:
                mid_pref = (first_step_in_ep + last_step_in_ep) // 2
                tail_pref = max(first_step_in_ep, last_step_in_ep - int(args.backfill_tail_offset))
                forbid: List[int] = []
                append_if_valid(mid_pref, forbid, backfills)
                append_if_valid(tail_pref, [k["step_keystep"] for k in backfills], backfills)
            elif len(selected_steps) == 1:
                k0 = selected_steps[0]
                mid = (first_step_in_ep + last_step_in_ep) // 2
                forbid = [k0]
                if k0 <= mid:
                    append_if_valid(max(first_step_in_ep, last_step_in_ep - int(args.backfill_tail_offset)), forbid[:], backfills)
                else:
                    append_if_valid((first_step_in_ep + k0) // 2, forbid[:], backfills)

            all_candidates.extend(arm_keysteps + backfills)

        all_keysteps = sim_utils.merge_keysteps_by_step(all_candidates)
        for item in all_keysteps:
            idx_before = item["idx_before"]
            idx_ks = item["idx_keystep"]
            action_before = list(map(float, np.asarray(frames[idx_before]["action"]).tolist()))
            action_ks = list(map(float, np.asarray(frames[idx_ks]["action"]).tolist()))
            manifest_rows.append({
                "episode_id": ep_id,
                "episode_name": ep_name,
                "keystep_type": item["type"],
                "change_point": int(item.get("cp", -1)),
                "idx_first": int(first_idx_global),
                "idx_before": int(idx_before),
                "idx_keystep": int(idx_ks),
                "step_first": int(first_step),
                "step_before": int(item["step_before"]),
                "step_keystep": int(item["step_keystep"]),
                "gripper_before": int(item["gripper_before"]),
                "gripper_after": int(item["gripper_after"]),
                "action_first": json.dumps(action_first),
                "action_before": json.dumps(action_before),
                "action_keystep": json.dumps(action_ks),
                "pos_delta_keystep": float(item["pos_delta_keystep"]),
                "rot_delta_keystep": float(item["rot_delta_keystep"]),
                "lookahead_used": int(item["lookahead_used"]),
                "binarize_mode": item["binarize_mode"],
                "found_premove": bool(item.get("found_premove", False)),
                "flip_enter_pos": float(item.get("flip_enter_pos", 0.0)),
                "flip_enter_rot": float(item.get("flip_enter_rot", 0.0)),
                "still_pos_thr": float(item.get("still_pos_thr", 0.0)),
                "still_rot_thr": float(item.get("still_rot_thr", 0.0)),
                "event_arm": item.get("event_arm", "single"),
            })

    df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.out_dir, "triplets_manifest.csv")
    keysteps_path = os.path.join(args.out_dir, "keysteps.csv")
    df.to_csv(manifest_path, index=False)
    df.to_csv(keysteps_path, index=False)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "script": "extract_real_robot_sim.py",
            "dataset": args.dataset,
            "episodes_detected": len(grouped),
            "triplets": int(len(df)),
            "out_root": args.out_dir,
            "manifest_csv": manifest_path,
            "keysteps_csv": keysteps_path,
            "test_mode": bool(args.test),
            "params": vars(args),
        }, f, ensure_ascii=False, indent=2)

    sim_utils.log(f"CSV in: {manifest_path}")
    sim_utils.log(f"CSV in: {keysteps_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
