"""Roll out a trained LeRobot Diffusion Policy in PhysicsBench's own sim env.

Tier-2 eval. Wraps a LeRobot DP checkpoint behind PhysicsBench's
`PolicyInterface(act/reset)` and runs episodes via `pb.make(task_id, ...)`.

PhysicsBench is not installed as a package here — it's vendored under
`PhysicsBench/`. The script adds that path to `sys.path` automatically
(or honor PYTHONPATH if you've already set it).

Usage:
    uv run python scripts/eval_dp_in_physicsbench.py \\
        --policy-path outputs/train/dp_edge_slide_pilot100/checkpoints/last/pretrained_model \\
        --task-id pb-pr-edge-slide-v1 \\
        --n-episodes 50 --seed-offset 1000 \\
        --output outputs/eval/dp_pilot100/results.json

Output JSON shape:
    {
        "policy_path": "...",
        "task_id": "...",
        "n_episodes": 50,
        "control_freq": 25,
        "aggregated": {
            "mean_score": ..., "std_score": ..., "success_rate": ...,
            "mean_length": ..., "n_episodes": ...,
            "wall_s": ..., "wall_s_per_episode": ...
        },
        "per_episode": [{"ep": i, "seed": s, "score": ..., "length": ...,
                         "terminated": bool, "truncated": bool}, ...]
    }

Source-key mapping (PhysicsBench obs → LeRobot dataset key):
    robot0_eef_pos      ┐
    robot0_eef_quat     ├──→  observation.state    (concat, 9-dim)
    robot0_gripper_qpos ┘
    rgb_ego        →  observation.images.ego
    rgb_exo_left   →  observation.images.exo_left
    rgb_exo_right  →  observation.images.exo_right
    rgb_gripper    →  observation.images.gripper
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Vendored PhysicsBench import — add its dir to sys.path if not already present.
# Deferred to run-time so `--help` works without the heavy robosuite stack installed.
_PB_PATH = Path(__file__).resolve().parent.parent / "PhysicsBench"


def _import_physicsbench():
    if _PB_PATH.is_dir() and str(_PB_PATH) not in sys.path:
        sys.path.insert(0, str(_PB_PATH))
    import physicsbench as pb  # noqa: E402
    return pb


from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402

# PhysicsBench obs keys ↔ LeRobot dataset image keys.
#
# base_task._build_obs_from_config stores frames under `rgb_<label>` where
# `<label>` is the full YAML camera key (e.g. `ego_cam`, `exo_front_cam`), so
# the obs dict keys end up as `rgb_ego_cam`, `rgb_exo_front_cam`, etc. The
# conversion script strips the `_cam` suffix when renaming to LeRobot dataset
# keys, giving e.g. `observation.images.ego`, `observation.images.exo_front`.
#
# Different tasks have different camera layouts (edge-slide uses ego, domino
# uses exo_front, etc.), so we build the per-policy mapping at adapter init
# time from the policy's expected image_features rather than hardcoding it.

_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _derive_cam_map(image_features: "list[str] | dict") -> dict[str, str]:
    """Build PhysicsBench-obs-key → LeRobot-dataset-key map from policy config.

    Each expected key ``observation.images.<label>`` is matched to PhysicsBench
    obs key ``rgb_<label>_cam`` (the inverse of what the converter does).
    """
    keys = list(image_features.keys()) if isinstance(image_features, dict) else list(image_features)
    cam_map: dict[str, str] = {}
    for dst_key in keys:
        if not dst_key.startswith("observation.images."):
            continue
        label = dst_key.removeprefix("observation.images.")
        cam_map[f"rgb_{label}_cam"] = dst_key
    if not cam_map:
        raise ValueError(
            f"Could not derive camera map from policy image_features: {keys!r}"
        )
    return cam_map


def _state_from_obs(obs: dict) -> np.ndarray:
    """Concat eef_pos + eef_quat + gripper_qpos → 9-dim state."""
    parts = []
    for k in _STATE_KEYS:
        if k not in obs:
            raise KeyError(f"Missing required proprio key in PhysicsBench obs: {k}")
        parts.append(np.asarray(obs[k], dtype=np.float32).ravel())
    return np.concatenate(parts)


def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 → CHW float32 in [0, 1]."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    # PhysicsBench applies `img[::-1]` (OpenGL vertical flip) before storing in
    # obs, leaving negative strides that torch.from_numpy refuses. Force a
    # contiguous copy first.
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0


def _build_batch(obs: dict, task_text: str, device: str, cam_map: dict[str, str]) -> dict:
    batch: dict = {
        "observation.state": torch.from_numpy(_state_from_obs(obs)).unsqueeze(0).to(device),
        "task": [task_text],
    }
    for src_key, dst_key in cam_map.items():
        if src_key not in obs:
            raise KeyError(
                f"Camera key {src_key!r} expected by policy not present in PhysicsBench obs. "
                f"Available rgb keys: {[k for k in obs if k.startswith('rgb_')]}"
            )
        batch[dst_key] = _img_to_tensor(obs[src_key]).unsqueeze(0).to(device)
    return batch


class LeRobotDPAdapter:
    """Thin wrapper that exposes a LeRobot DP checkpoint via the act/reset API."""

    def __init__(self, policy_path: str, device: str, task_text: str):
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.device = device
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(policy_path, config=cfg).to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=policy_path,
        )
        self.device = device
        self.task_text = task_text
        # Derive per-policy camera map from the loaded config (different tasks
        # have different camera sets — e.g. edge-slide has 'ego', domino-select
        # has 'exo_front', etc.).
        self.cam_map = _derive_cam_map(self.policy.config.image_features)
        print(f"[adapter] camera map: {self.cam_map}")

    def reset(self) -> None:
        self.policy.reset()

    @torch.inference_mode()
    def act(self, obs: dict) -> np.ndarray:
        batch = _build_batch(obs, self.task_text, self.device, self.cam_map)
        batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.squeeze(0).cpu().numpy().astype(np.float32)


def run_episodes(
    adapter: LeRobotDPAdapter,
    task_id: str,
    seeds: list[int],
    control_freq: int,
    max_episode_steps: int,
    videos_dir: Path | None = None,
    video_camera: str = "rgb_ego_cam",
) -> dict:
    pb = _import_physicsbench()
    config_overrides = {"control_freq": control_freq}
    env = pb.make(
        task_id,
        obs_mode="combo",
        render_mode=None,
        config_overrides=config_overrides,
        max_episode_steps=max_episode_steps,
    )

    if videos_dir is not None:
        videos_dir.mkdir(parents=True, exist_ok=True)
        from lerobot.utils.io_utils import write_video

    n_episodes = len(seeds)
    per_episode = []
    scores: list[float] = []
    lengths: list[int] = []
    start = time.time()

    for ep, seed in enumerate(seeds):
        obs, _info = env.reset(seed=seed)
        adapter.reset()

        # Buffer the chosen camera if we're saving video. PhysicsBench frames
        # come back with negative strides (OpenGL flip); copy into contiguous
        # arrays so the eventual stack/encode is straightforward.
        frames: list[np.ndarray] | None = [] if videos_dir is not None else None
        if frames is not None and video_camera in obs:
            frames.append(np.ascontiguousarray(obs[video_camera]))

        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action = adapter.act(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            if frames is not None and video_camera in obs:
                frames.append(np.ascontiguousarray(obs[video_camera]))

        score = float(info.get("task_score", 0.0))
        length = int(info.get("step", -1))
        scores.append(score)
        lengths.append(length)

        video_path: str | None = None
        if videos_dir is not None and frames:
            outcome = "success" if score > 0.9 else "fail"
            video_path = str(videos_dir / f"ep{ep:03d}_seed{seed}_{outcome}_len{length}.mp4")
            write_video(video_path, frames, fps=control_freq)

        per_episode.append({
            "ep": ep,
            "seed": seed,
            "score": score,
            "length": length,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            **({"video_path": video_path} if video_path else {}),
        })
        elapsed = time.time() - start
        print(
            f"[rollout] ep {ep + 1:3d}/{n_episodes}  seed={seed}  score={score:.3f}  "
            f"len={length}  cum_succ={(np.array(scores) > 0.9).mean():.2%}  "
            f"elapsed={elapsed:.1f}s"
            + (f"  video={Path(video_path).name}" if video_path else "")
        )

    env.close()

    scores_arr = np.array(scores, dtype=np.float64)
    lengths_arr = np.array(lengths, dtype=np.float64)
    wall = time.time() - start
    return {
        "aggregated": {
            "mean_score": float(scores_arr.mean()) if len(scores_arr) else float("nan"),
            "std_score": float(scores_arr.std()) if len(scores_arr) else float("nan"),
            "success_rate": float((scores_arr > 0.9).mean()) if len(scores_arr) else float("nan"),
            "mean_length": float(lengths_arr.mean()) if len(lengths_arr) else float("nan"),
            "n_episodes": int(n_episodes),
            "wall_s": wall,
            "wall_s_per_episode": wall / max(1, n_episodes),
        },
        "per_episode": per_episode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="Local checkpoint dir or HF id.")
    parser.add_argument("--task-id", required=True, help="PhysicsBench task id (e.g. pb-pr-edge-slide-v1).")
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Number of episodes (ignored if --seeds is given).")
    parser.add_argument("--seed-offset", type=int, default=1000,
                        help="First seed; ep i uses seed_offset+i. Ignored if --seeds is given.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated explicit seed list (e.g. '1001,1028,1034,1044,1049'). "
        "Overrides --n-episodes/--seed-offset; useful for re-running specific episodes "
        "(e.g. for video capture).",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Directory for per-episode mp4s of the selected camera. Off by default. "
        "Files: ep<idx>_seed<seed>_<success|fail>_len<N>.mp4",
    )
    parser.add_argument(
        "--video-camera",
        type=str,
        default="rgb_ego_cam",
        help="Which PhysicsBench camera key to record (rgb_ego_cam, rgb_exo_left_cam, "
        "rgb_exo_right_cam, rgb_gripper_cam).",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=25,
        help="Override env control frequency. 25 matches the dataset's 25 Hz subsample (recommended). "
        "100 is the PhysicsBench default — use only with a per-step action repeat strategy "
        "(not implemented here).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Cap episode length. At control_freq=25, 1000 steps = 40 sim-seconds — generous for edge-slide.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--task-text", default=None, help="Language task description for VLA-style policies. "
                        "Defaults to the dataset's stored task text via the registry; safe to leave unset for DP.")
    parser.add_argument("--output", required=True, help="Where to write results.json.")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed_offset + i for i in range(args.n_episodes)]

    print(f"[rollout] policy: {args.policy_path}")
    print(f"[rollout] task: {args.task_id} (control_freq={args.control_freq}, max_ep_steps={args.max_episode_steps})")
    print(f"[rollout] n_episodes={len(seeds)}, seeds={seeds[:5]}{'...' if len(seeds) > 5 else ''}")
    if args.videos_dir:
        print(f"[rollout] saving videos ({args.video_camera}) to {args.videos_dir}")

    adapter = LeRobotDPAdapter(
        policy_path=args.policy_path,
        device=args.device,
        task_text=args.task_text or args.task_id,
    )

    results = run_episodes(
        adapter=adapter,
        task_id=args.task_id,
        seeds=seeds,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
        videos_dir=args.videos_dir,
        video_camera=args.video_camera,
    )

    out = {
        "policy_path": args.policy_path,
        "task_id": args.task_id,
        "n_episodes": len(seeds),
        "control_freq": args.control_freq,
        "seeds": seeds,
        **results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    agg = results["aggregated"]
    print(
        f"\n[rollout] {args.task_id}: "
        f"mean_score={agg['mean_score']:.3f} ± {agg['std_score']:.3f}  "
        f"success_rate={agg['success_rate']:.2%}  "
        f"mean_length={agg['mean_length']:.1f}  "
        f"({agg['wall_s_per_episode']:.1f}s/ep)"
    )
    print(f"[rollout] wrote {out_path}")


if __name__ == "__main__":
    main()
