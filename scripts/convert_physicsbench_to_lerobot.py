"""Convert a PhysicsBench-format dataset to a LeRobot-policy-ready local dataset.

Auto-detects from source ``info.json``:
  - Cameras: every feature with ``dtype="video"``. ``observation.rgb_<label>_cam``
    is renamed to ``observation.images.<label>``; other shapes fall back to
    ``observation.images.<sanitized>``.
  - Arms: scans for ``observation.robot{N}_eef_pos`` keys.
  - Per-arm gripper width: read from feature shape (some grippers have 2 dofs,
    some 1 — e.g. suction).
  - Action dim and task description: read from source.

Builds ``observation.state`` = concat over arms of ``[eef_pos, eef_quat,
gripper_qpos]``. Optionally append force/torque with ``--include-ft``.

Subsamples 100 Hz proprio rows to the camera frame rate (default 25 Hz) using
the first video key's ``frame_index.<key>`` column. PhysicsBench renders cameras
at 25 Hz with frame caching between renders, so this is lossless.

Privileged sim state (``observation.object_*``, ``cube_*``, etc.) is dropped via
the whitelist approach (we only KEEP what's explicitly built).

Usage:
    # Single-arm edge-slide, 10 episodes for laptop sanity check
    uv run python scripts/convert_physicsbench_to_lerobot.py \\
      --src-repo-id manav-robotics/pb-pr-edge-slide-v1 \\
      --num-episodes 10

    # Bimanual task, full conversion, include F/T in state
    uv run python scripts/convert_physicsbench_to_lerobot.py \\
      --src-repo-id manav-robotics/pb-hetbi-bowl-table-v1 \\
      --num-episodes -1 \\
      --include-ft
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchcodec.decoders import VideoDecoder

from lerobot.datasets.lerobot_dataset import LeRobotDataset

HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"


# ── source resolution ───────────────────────────────────────────────────────


def resolve_src_path(repo_id: str | None, src_path: Path | None) -> Path:
    """Resolve source snapshot dir from a repo_id (via HF Hub cache) or direct path."""
    if src_path is not None:
        if not src_path.is_dir():
            raise FileNotFoundError(f"--src-path does not exist: {src_path}")
        return src_path
    if repo_id is None:
        raise ValueError("Provide either --src-repo-id or --src-path")
    cache_dir = HF_HUB_CACHE / f"datasets--{repo_id.replace('/', '--')}"
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"{cache_dir} not found. Run `hf download --repo-type=dataset {repo_id}` first."
        )
    main_ref = cache_dir / "refs" / "main"
    if main_ref.is_file():
        sha = main_ref.read_text().strip()
        return cache_dir / "snapshots" / sha
    snapshots = sorted((cache_dir / "snapshots").iterdir())
    if len(snapshots) != 1:
        raise ValueError(
            f"Multiple snapshots at {cache_dir}/snapshots; pass --src-path explicitly."
        )
    return snapshots[0]


# ── feature detection ───────────────────────────────────────────────────────


_CAM_NAME_RE = re.compile(r"^observation\.rgb_(?P<label>.+)_cam$")
_ARM_RE = re.compile(r"^observation\.robot(?P<n>\d+)_eef_pos$")


def detect_cameras(features: dict) -> dict[str, str]:
    """Map source video keys → destination ``observation.images.<label>`` keys."""
    out: dict[str, str] = {}
    for key, ft in features.items():
        if ft.get("dtype") != "video":
            continue
        m = _CAM_NAME_RE.match(key)
        if m:
            label = m.group("label")
        else:
            stem = key.removeprefix("observation.")
            label = re.sub(r"[^A-Za-z0-9_]", "_", stem)
        out[key] = f"observation.images.{label}"
    if not out:
        raise ValueError("No video features found in source dataset.")
    return out


def detect_arms(features: dict) -> list[int]:
    """Find arm indices N such that observation.robot{N}_eef_pos exists."""
    arms: set[int] = set()
    for key in features:
        m = _ARM_RE.match(key)
        if m:
            arms.add(int(m.group("n")))
    if not arms:
        raise ValueError(
            "No arms detected (no observation.robotN_eef_pos features). "
            "This script assumes PhysicsBench proprio naming."
        )
    return sorted(arms)


def state_recipe(features: dict, arms: list[int], include_ft: bool) -> tuple[list[str], list[str]]:
    """Return (per-element source columns, human-readable names) for observation.state."""
    cols: list[str] = []
    names: list[str] = []
    multi_arm = len(arms) > 1
    for n in arms:
        prefix = f"r{n}_" if multi_arm else ""
        for sub in ("eef_pos", "eef_quat", "gripper_qpos"):
            base = f"observation.robot{n}_{sub}"
            if base not in features:
                raise ValueError(f"Missing required state feature: {base}")
            dim = features[base]["shape"][0]
            for i in range(dim):
                cols.append(f"{base}.{i}")
            if sub == "eef_pos":
                names.extend([f"{prefix}eef_x", f"{prefix}eef_y", f"{prefix}eef_z"])
            elif sub == "eef_quat":
                names.extend([f"{prefix}eef_qx", f"{prefix}eef_qy", f"{prefix}eef_qz", f"{prefix}eef_qw"])
            else:
                names.extend([f"{prefix}grip_{i}" for i in range(dim)])
    if include_ft:
        ft_key = "observation.force_torque"
        if ft_key not in features:
            raise ValueError("--include-ft requested but observation.force_torque missing in source.")
        dim = features[ft_key]["shape"][0]
        cols.extend([f"{ft_key}.{i}" for i in range(dim)])
        names.extend([f"ft_{ax}" for ax in ("fx", "fy", "fz", "tx", "ty", "tz")][:dim])
    return cols, names


def action_columns(features: dict) -> tuple[list[str], list[str]]:
    """Per-element action columns + human names. PhysicsBench layout: [dx,dy,dz,droll,dpitch,dyaw,gripper] per arm."""
    if "action" not in features:
        raise ValueError("Source missing 'action' feature.")
    dim = features["action"]["shape"][0]
    cols = [f"action.{i}" for i in range(dim)]
    if dim % 7 == 0:
        per_arm = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
        n_arms = dim // 7
        names = [
            (f"r{a}_" if n_arms > 1 else "") + name
            for a in range(n_arms)
            for name in per_arm
        ]
    else:
        names = [f"a{i}" for i in range(dim)]
    return cols, names


# ── camera reader ───────────────────────────────────────────────────────────


class CameraReader:
    """Lazy per-mp4 VideoDecoder cache."""

    def __init__(self, src: Path):
        self.src = src
        self._cache: dict[str, VideoDecoder] = {}

    def get(self, src_cam_key: str, chunk_index: int, file_index: int) -> VideoDecoder:
        path = (
            self.src / "videos" / src_cam_key
            / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
        )
        key = str(path)
        if key not in self._cache:
            self._cache[key] = VideoDecoder(str(path), seek_mode="approximate")
        return self._cache[key]

    def read_frame(self, src_cam_key: str, chunk_index: int, file_index: int, frame_idx: int) -> np.ndarray:
        decoder = self.get(src_cam_key, chunk_index, file_index)
        frame = decoder.get_frame_at(index=int(frame_idx)).data
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)
        return frame.permute(1, 2, 0).contiguous().numpy()


# ── main flow ───────────────────────────────────────────────────────────────


def load_episode_meta(src: Path) -> pd.DataFrame:
    """Load and concatenate all episode metadata parquets, sorted by episode_index."""
    files = sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {src}/meta/episodes/")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def build_features_dict(
    src_features: dict,
    cam_rename: dict[str, str],
    state_dim: int,
    state_names: list[str],
    action_dim: int,
    action_names: list[str],
) -> dict:
    sample_video = next(iter(cam_rename))
    h, w, c = src_features[sample_video]["shape"]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
        **{
            new_key: {
                "dtype": "video",
                "shape": (
                    src_features[old_key]["shape"][0],
                    src_features[old_key]["shape"][1],
                    src_features[old_key]["shape"][2],
                ),
                "names": ["height", "width", "channels"],
            }
            for old_key, new_key in cam_rename.items()
        },
    }


def convert(
    src: Path,
    dst_repo_id: str,
    num_episodes: int,
    include_ft: bool,
    overwrite: bool,
) -> None:
    src_info = json.loads((src / "meta" / "info.json").read_text())
    src_features = src_info["features"]

    cam_rename = detect_cameras(src_features)
    arms = detect_arms(src_features)
    state_cols, state_names = state_recipe(src_features, arms, include_ft)
    action_cols, action_names = action_columns(src_features)

    # Pick the first video key's frame_index column as the camera-rate reference.
    first_cam_src = next(iter(cam_rename))
    cam_rate_col = f"frame_index.{first_cam_src.removeprefix('observation.')}"
    if cam_rate_col not in {*src_features.keys(), cam_rate_col}:
        # The frame_index features are stored as integer features; verify by
        # checking if any matching column exists in the parquet schema later.
        pass  # we'll catch it when reading parquet
    cam_fps = src_features[first_cam_src].get("fps")
    if cam_fps is None:
        raise ValueError(f"Source camera feature {first_cam_src} missing fps.")

    print(f"[convert] src: {src}")
    print(f"[convert] arms detected: {arms}")
    print(f"[convert] cameras: {cam_rename}")
    print(f"[convert] state dim: {len(state_cols)}{' (includes F/T)' if include_ft else ''}")
    print(f"[convert] action dim: {len(action_cols)}")
    print(f"[convert] target fps: {cam_fps}")

    dst_root = LEROBOT_CACHE / dst_repo_id
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst_root} already exists. Pass --overwrite or pick a different --dst-repo-id."
            )
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)

    ep_meta = load_episode_meta(src)
    n_total = len(ep_meta)
    n = n_total if num_episodes < 0 else min(num_episodes, n_total)
    print(f"[convert] episodes available: {n_total}, processing: {n}")

    task_description = (
        src_info.get("task_config", {}).get("description")
        or src_info.get("task_id")
        or "task"
    )
    print(f"[convert] task description: {task_description!r}")

    features = build_features_dict(
        src_features, cam_rename, len(state_cols), state_names, len(action_cols), action_names
    )

    new_ds = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=int(cam_fps),
        features=features,
        robot_type=src_info.get("robot_type"),
        use_videos=True,
    )
    print(f"[convert] writing to: {new_ds.root}")

    cams = CameraReader(src)
    needed_cols = [cam_rate_col] + state_cols + action_cols
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}

    for ep_idx in range(n):
        ep_row = ep_meta.iloc[ep_idx]
        data_chunk = int(ep_row["data/chunk_index"])
        data_file = int(ep_row["data/file_index"])
        from_idx = int(ep_row["dataset_from_index"])
        to_idx = int(ep_row["dataset_to_index"])

        if (data_chunk, data_file) not in data_cache:
            data_cache[(data_chunk, data_file)] = pd.read_parquet(
                src / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet",
                columns=needed_cols,
            )
        df = data_cache[(data_chunk, data_file)].iloc[from_idx:to_idx]
        sub = df[df[cam_rate_col].notna()].reset_index(drop=True)

        cam_locations = {
            src_key: (
                int(ep_row[f"videos/{src_key}/chunk_index"]),
                int(ep_row[f"videos/{src_key}/file_index"]),
            )
            for src_key in cam_rename
        }

        state_arr = sub[state_cols].to_numpy(dtype=np.float32)
        action_arr = sub[action_cols].to_numpy(dtype=np.float32)
        cam_indices = sub[cam_rate_col].to_numpy(dtype=np.int64)

        for i in range(len(sub)):
            frame: dict = {
                "task": task_description,
                "observation.state": state_arr[i],
                "action": action_arr[i],
            }
            for src_key, new_key in cam_rename.items():
                ck, fk = cam_locations[src_key]
                frame[new_key] = cams.read_frame(src_key, ck, fk, cam_indices[i])
            new_ds.add_frame(frame)

        new_ds.save_episode()
        print(f"[convert] ep {ep_idx + 1}/{n}: wrote {len(sub)} frames (orig length {len(df)} @ source rate)")

    new_ds.finalize()
    print(f"[convert] done. dataset at {new_ds.root}")
    print(f"[convert] total frames: {new_ds.meta.total_frames}, episodes: {new_ds.meta.total_episodes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--src-repo-id",
        help="HF repo id (e.g. manav-robotics/pb-pr-edge-slide-v1). "
        "Resolved via the HF Hub cache. Required unless --src-path is given.",
    )
    src_group.add_argument(
        "--src-path",
        type=Path,
        help="Direct path to the source snapshot dir. Use when not in HF Hub cache.",
    )
    parser.add_argument(
        "--dst-repo-id",
        help="Destination repo id under $HF_LEROBOT_HOME. "
        "Defaults to '<src-repo-id>-lerobot' (with '-ft' suffix if --include-ft).",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Episodes to convert. -1 for all.")
    parser.add_argument(
        "--include-ft",
        action="store_true",
        help="Append observation.force_torque to observation.state (raises state dim).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the destination dir if it already exists.",
    )
    args = parser.parse_args()

    src = resolve_src_path(args.src_repo_id, args.src_path)
    if args.dst_repo_id:
        dst_repo_id = args.dst_repo_id
    elif args.src_repo_id:
        suffix = "-lerobot-ft" if args.include_ft else "-lerobot"
        dst_repo_id = args.src_repo_id + suffix
    else:
        raise ValueError("Provide --dst-repo-id when using --src-path.")

    convert(src, dst_repo_id, args.num_episodes, args.include_ft, args.overwrite)


if __name__ == "__main__":
    main()
