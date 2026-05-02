"""Action-MSE eval for a Diffusion Policy checkpoint against a held-out LeRobot dataset.

Tier-1 screen before paying for sim rollouts: how well does the policy
reproduce the ground-truth actions on episodes it didn't see at training time?

Usage:
    uv run python scripts/eval_dp_action_mse.py \\
        --policy-path outputs/train/dp_edge_slide_pilot100/checkpoints/last/pretrained_model \\
        --val-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot-val30 \\
        --output outputs/eval/dp_pilot100/action_mse.json

Output JSON shape:
    {
        "policy_path": "...",
        "val_repo_id": "...",
        "n_frames": <int>,
        "n_episodes": <int>,
        "action_dim": <int>,
        "mse_overall": <float>,
        "mse_per_dim": [<float>, ...],
        "rmse_per_dim": [<float>, ...],
        "action_std_per_dim": [<float>, ...],
        "nrmse_per_dim": [<float>, ...]   # rmse / action_std, scale-free
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors


def _load_policy_and_processors(policy_path: str, device: str):
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg)
    policy = policy.to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=policy_path)
    return policy, pre, post, cfg


def _frame_to_batch(frame: dict, device: str) -> dict:
    """Convert a LeRobotDataset frame (single timestep) to a batched policy input.

    Adds a leading batch dim, moves tensors to device. Strings (e.g. ``task``)
    pass through wrapped in a list.
    """
    out: dict = {}
    for k, v in frame.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, str):
            out[k] = [v]
        # silently drop unknown types (e.g. ints like episode_index)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="Local checkpoint dir or HF id.")
    parser.add_argument("--val-repo-id", required=True, help="Held-out LeRobot dataset repo id.")
    parser.add_argument("--output", required=True, help="Where to write the action_mse.json.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Cap evaluation to N frames (debug). -1 = all.",
    )
    args = parser.parse_args()

    print(f"[mse] loading policy from {args.policy_path}")
    policy, preprocessor, postprocessor, _ = _load_policy_and_processors(args.policy_path, args.device)

    print(f"[mse] loading val dataset {args.val_repo_id}")
    ds = LeRobotDataset(repo_id=args.val_repo_id)
    n_frames = len(ds) if args.max_frames < 0 else min(args.max_frames, len(ds))
    print(f"[mse] frames to evaluate: {n_frames} / {len(ds)} (episodes: {ds.num_episodes})")

    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    last_episode_idx = -1

    with torch.inference_mode():
        for i in tqdm(range(n_frames), desc="action MSE"):
            frame = ds[i]
            ep_idx = int(frame.get("episode_index", -1))
            if ep_idx != last_episode_idx:
                # New episode: clear DP's rolling action queue.
                policy.reset()
                last_episode_idx = ep_idx

            gt_action = frame["action"].cpu().numpy()  # (action_dim,)

            batch = _frame_to_batch(frame, args.device)
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)
            pred_action = action.squeeze(0).cpu().numpy()

            preds.append(pred_action)
            truths.append(gt_action)

    preds_arr = np.stack(preds)   # (N, action_dim)
    truths_arr = np.stack(truths)
    action_dim = preds_arr.shape[1]

    diff = preds_arr - truths_arr
    mse_per_dim = (diff ** 2).mean(axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    action_std_per_dim = truths_arr.std(axis=0) + 1e-12
    nrmse_per_dim = rmse_per_dim / action_std_per_dim

    summary = {
        "policy_path": args.policy_path,
        "val_repo_id": args.val_repo_id,
        "n_frames": int(preds_arr.shape[0]),
        "n_episodes": int(ds.num_episodes),
        "action_dim": int(action_dim),
        "mse_overall": float(mse_per_dim.mean()),
        "mse_per_dim": mse_per_dim.tolist(),
        "rmse_per_dim": rmse_per_dim.tolist(),
        "action_std_per_dim": action_std_per_dim.tolist(),
        "nrmse_per_dim": nrmse_per_dim.tolist(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"[mse] overall MSE: {summary['mse_overall']:.6f}")
    print(f"[mse] per-dim NRMSE (rmse / action_std):")
    for i, n in enumerate(nrmse_per_dim):
        print(f"        dim {i}: {n:.3f}")
    print(f"[mse] wrote {out_path}")


if __name__ == "__main__":
    main()
