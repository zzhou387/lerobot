"""Convert a PhysicsBench-format dataset to a LeRobot-policy-ready local dataset.

Supports both PhysicsBench codebase versions:

- **v2.1**: per-episode parquet (``data/chunk-XXX/episode_<ID>.parquet``),
  per-episode mp4 per camera (``videos/chunk-XXX/<cam>/episode_<ID>.mp4``),
  per-episode metadata JSONs (``meta/episode_<ID>_metadata.json``) carrying
  ``annotation_rating`` (4 = success-with-recovery, 5 = clean single-shot).
- **v3.0**: episode meta parquet under ``meta/episodes/`` listing positional
  ranges into shared multi-episode parquets / mp4s.

Auto-detects from source ``info.json`` (in either layout):
  - Cameras: every feature with ``dtype="video"``. ``observation.rgb_<label>_cam``
    is renamed to ``observation.images.<label>``; other shapes fall back to
    ``observation.images.<sanitized>``.
  - Arms: scans for ``observation.robot{N}_eef_pos`` keys.
  - Per-arm gripper width: read from feature shape (some grippers have 2 dofs,
    some 1 — e.g. suction).
  - Action dim and task description: read from source.

For v2.1 only, ``--min-rating N`` filters episodes by ``annotation_rating``
(default 5: keep only clean single-shot demos; pass 4 to also include
recovery-style demos).

Builds ``observation.state`` = concat over arms of ``[eef_pos, eef_quat,
gripper_qpos]``. Optionally append force/torque with ``--include-ft``.

Subsamples 100 Hz proprio rows to the camera frame rate (default 25 Hz) using
the first video key's ``frame_index.<key>`` column. PhysicsBench renders cameras
at 25 Hz with frame caching between renders, so this is lossless.

Privileged sim state (``observation.object_*``, etc.) is dropped via the
whitelist approach (we only KEEP what's explicitly built).

Usage:
    # v2.1 single-arm edge-slide, 10 rating-5 episodes, laptop sanity check
    uv run python scripts/convert_physicsbench_to_lerobot.py \\
      --src-path ~/manav/dataset/pb-pr-edge-slide-v1 \\
      --dst-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot \\
      --num-episodes 10

    # v2.1, full conversion, include rating-4 (recovery) episodes too
    uv run python scripts/convert_physicsbench_to_lerobot.py \\
      --src-path ~/manav/dataset/pb-pr-edge-slide-v1 \\
      --dst-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot \\
      --num-episodes -1 --min-rating 4

    # v3.0 from HF Hub cache (legacy path)
    uv run python scripts/convert_physicsbench_to_lerobot.py \\
      --src-repo-id manav-robotics/pb-pr-edge-slide-v1 \\
      --num-episodes 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchcodec.decoders import VideoDecoder

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME as LEROBOT_CACHE

HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


# ── source resolution ───────────────────────────────────────────────────────


def resolve_src_path(repo_id: str | None, src_path: Path | None) -> Path:
    """Resolve source dataset dir from a repo_id (HF Hub cache) or direct path."""
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


# ── feature detection (shared across versions) ─────────────────────────────


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


def infer_features_from_parquet(parquet_path: Path) -> dict:
    """Group parquet columns of the form ``<base>.<i>`` into ``{base: {shape: [N]}}``.

    v2.1 info.json doesn't list proprio/action features; they're only implied by
    the per-element columns in the data parquet. We read the parquet schema once
    and reconstruct the shapes.
    """
    schema = pd.read_parquet(parquet_path, columns=None).head(0)
    shapes: dict[str, int] = {}
    for col in schema.columns:
        if "." in col:
            base, suffix = col.rsplit(".", 1)
            if suffix.isdigit():
                shapes[base] = max(shapes.get(base, 0), int(suffix) + 1)
    return {
        base: {"dtype": "float32", "shape": [size]}
        for base, size in shapes.items()
    }


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
    """Per-element action columns + human names. PhysicsBench: [dx,dy,dz,droll,dpitch,dyaw,gripper] per arm."""
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


# ── per-episode read result, version-agnostic ───────────────────────────────


@dataclass
class EpisodePayload:
    """Output of a per-version reader: everything needed to write one episode."""
    episode_id: str
    sub_df: pd.DataFrame              # rows already subsampled to camera rate
    cam_indices: np.ndarray           # per-row frame-index into each cam's mp4
    cam_video_paths: dict[str, Path]  # source video file path keyed by source cam key
    task_text: str


def _read_episode_rows(parquet_path: Path, cam_rate_col: str, needed_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Load a parquet, subsample to camera rate, return (sub_df, full_len)."""
    df = pd.read_parquet(parquet_path, columns=needed_cols)
    full_len = len(df)
    sub = df[df[cam_rate_col].notna()].reset_index(drop=True)
    return sub, full_len


# ── v2.1 reader ─────────────────────────────────────────────────────────────


def _v21_extract_id(meta_path: Path) -> str:
    """`meta/episode_000074_metadata.json` → '000074'."""
    return meta_path.stem.removeprefix("episode_").removesuffix("_metadata")


def _v21_episode_chunk_dir(src: Path, episode_id: str, kind: str, cam: str | None = None) -> Path:
    """Find which `chunk-XXX/` holds this episode's data/video. Falls back to globbing."""
    if kind == "data":
        candidates = sorted((src / "data").glob(f"chunk-*/episode_{episode_id}.parquet"))
    else:
        candidates = sorted((src / "videos").glob(f"chunk-*/{cam}/episode_{episode_id}.mp4"))
    if not candidates:
        raise FileNotFoundError(f"v2.1 {kind} file for episode {episode_id} not found under {src}")
    return candidates[0]


def iter_v21(
    src: Path,
    cam_rename: dict[str, str],
    cam_rate_col: str,
    needed_cols: list[str],
    min_rating: int,
    max_episodes: int,
    skip_episodes: int = 0,
):
    """Yield EpisodePayload for v2.1 datasets, filtered by annotation_rating.

    `skip_episodes` skips the first N rating-passing episodes before yielding
    any — useful for cleanly carving train/val splits from the same source
    (e.g. train on first 100, validate on the next 30).
    """
    meta_files = sorted((src / "meta").glob("episode_*_metadata.json"))
    if not meta_files:
        raise FileNotFoundError(f"No v2.1 episode metadata under {src}/meta/")

    kept = 0
    skipped = 0
    skipped_for_offset = 0
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text())
        rating = meta.get("annotation_rating")
        if rating is None or rating < min_rating:
            skipped += 1
            continue

        if skipped_for_offset < skip_episodes:
            skipped_for_offset += 1
            continue

        episode_id = _v21_extract_id(meta_path)
        parquet_path = _v21_episode_chunk_dir(src, episode_id, "data")
        sub_df, full_len = _read_episode_rows(parquet_path, cam_rate_col, needed_cols)

        video_paths = {
            src_key: _v21_episode_chunk_dir(src, episode_id, "video", cam=src_key.removeprefix("observation."))
            for src_key in cam_rename
        }

        task_text = (
            meta.get("task_name")
            or meta.get("task_id")
            or "task"
        )

        yield EpisodePayload(
            episode_id=episode_id,
            sub_df=sub_df,
            cam_indices=sub_df[cam_rate_col].to_numpy(dtype=np.int64),
            cam_video_paths=video_paths,
            task_text=task_text,
        ), full_len

        kept += 1
        if max_episodes > 0 and kept >= max_episodes:
            break

    print(
        f"[v21] kept {kept}, skipped {skipped} (rating < {min_rating}), "
        f"skipped {skipped_for_offset} rating-passing for --skip-episodes offset"
    )


# ── v3.0 reader ─────────────────────────────────────────────────────────────


def iter_v3(
    src: Path,
    cam_rename: dict[str, str],
    cam_rate_col: str,
    needed_cols: list[str],
    max_episodes: int,
):
    """Yield EpisodePayload for v3.0 datasets (shared multi-episode parquets/mp4s)."""
    files = sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No v3.0 episode metadata under {src}/meta/episodes/")
    ep_meta = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    ep_meta = ep_meta.sort_values("episode_index").reset_index(drop=True)

    n_total = len(ep_meta)
    n = n_total if max_episodes < 0 else min(max_episodes, n_total)
    print(f"[v3] episodes available: {n_total}, processing: {n}")

    parquet_cache: dict[tuple[int, int], pd.DataFrame] = {}

    for ep_idx in range(n):
        ep_row = ep_meta.iloc[ep_idx]
        data_chunk = int(ep_row["data/chunk_index"])
        data_file = int(ep_row["data/file_index"])
        from_idx = int(ep_row["dataset_from_index"])
        to_idx = int(ep_row["dataset_to_index"])

        if (data_chunk, data_file) not in parquet_cache:
            parquet_cache[(data_chunk, data_file)] = pd.read_parquet(
                src / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet",
                columns=needed_cols,
            )
        df = parquet_cache[(data_chunk, data_file)].iloc[from_idx:to_idx]
        full_len = len(df)
        sub = df[df[cam_rate_col].notna()].reset_index(drop=True)

        # In v3, cam frame indices are GLOBAL within the cam's shared mp4 file.
        # We resolve the file at read time instead of caching per-episode.
        video_paths = {
            src_key: (
                src / "videos" / src_key
                / f"chunk-{int(ep_row[f'videos/{src_key}/chunk_index']):03d}"
                / f"file-{int(ep_row[f'videos/{src_key}/file_index']):03d}.mp4"
            )
            for src_key in cam_rename
        }

        yield EpisodePayload(
            episode_id=str(int(ep_row["episode_index"])),
            sub_df=sub,
            cam_indices=sub[cam_rate_col].to_numpy(dtype=np.int64),
            cam_video_paths=video_paths,
            task_text="",  # filled in by caller (uses global task description from info.json)
        ), full_len


# ── camera reader ───────────────────────────────────────────────────────────


class VideoDecoderCache:
    """Lazy mp4 → VideoDecoder cache, keyed by file path."""

    def __init__(self):
        self._cache: dict[str, VideoDecoder] = {}

    def read_frame(self, path: Path, frame_idx: int) -> np.ndarray:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = VideoDecoder(key, seek_mode="approximate")
        frame = self._cache[key].get_frame_at(index=int(frame_idx)).data
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)
        return frame.permute(1, 2, 0).contiguous().numpy()

    def probe_shape(self, path: Path) -> tuple[int, int, int]:
        """Return (H, W, C) of the first frame of the given mp4."""
        key = str(path)
        if key not in self._cache:
            self._cache[key] = VideoDecoder(key, seek_mode="approximate")
        frame = self._cache[key].get_frame_at(index=0).data  # CHW
        return int(frame.shape[1]), int(frame.shape[2]), int(frame.shape[0])


# ── features dict construction ──────────────────────────────────────────────


def build_features_dict(
    cam_rename: dict[str, str],
    state_dim: int,
    state_names: list[str],
    action_dim: int,
    action_names: list[str],
    video_shape: tuple[int, int, int],
) -> dict:
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
                "shape": video_shape,
                "names": ["height", "width", "channels"],
            }
            for new_key in cam_rename.values()
        },
    }


# ── per-episode write (shared between single- and multi-process paths) ─────


def _write_episode(
    new_ds: LeRobotDataset,
    payload: "EpisodePayload",
    cam_rename: dict[str, str],
    state_cols: list[str],
    action_cols: list[str],
    global_task_text: str,
    decoders: VideoDecoderCache,
) -> int:
    """Push one episode's frames into ``new_ds`` and finalize that episode.

    Returns the number of frames written.
    """
    state_arr = payload.sub_df[state_cols].to_numpy(dtype=np.float32)
    action_arr = payload.sub_df[action_cols].to_numpy(dtype=np.float32)
    task_text = payload.task_text or global_task_text

    for i in range(len(payload.sub_df)):
        frame: dict = {
            "task": task_text,
            "observation.state": state_arr[i],
            "action": action_arr[i],
        }
        for src_key, new_key in cam_rename.items():
            frame[new_key] = decoders.read_frame(
                payload.cam_video_paths[src_key], payload.cam_indices[i]
            )
        new_ds.add_frame(frame)
    new_ds.save_episode()
    return len(payload.sub_df)


# ── multi-process worker ───────────────────────────────────────────────────


def _worker_convert_slice(
    worker_id: int,
    output_root: str,
    src: str,
    cam_rename: dict[str, str],
    cam_rate_col: str,
    needed_cols: list[str],
    state_cols: list[str],
    action_cols: list[str],
    cam_fps: int,
    features: dict,
    robot_type: str | None,
    global_task_text: str,
    min_rating: int,
    skip_episodes: int,
    num_episodes: int,
    codebase_version: str,
) -> tuple[str, int, int]:
    """Convert an episode slice into a fresh LeRobet dataset under ``output_root``.

    Args paths are passed as ``str`` (not ``Path``) because some multiprocessing
    start methods don't pickle ``Path`` cleanly across module versions.

    Returns ``(output_root, n_episodes_written, n_frames_written)``.
    """
    src_path = Path(src)
    output_path = Path(output_root)

    new_ds = LeRobotDataset.create(
        repo_id=f"local-staging/worker-{worker_id}",
        root=output_path,
        fps=int(cam_fps),
        features=features,
        robot_type=robot_type,
        use_videos=True,
    )
    decoders = VideoDecoderCache()

    if codebase_version.startswith("v2"):
        episode_iter = iter_v21(
            src_path, cam_rename, cam_rate_col, needed_cols, min_rating, num_episodes,
            skip_episodes=skip_episodes,
        )
    else:
        episode_iter = iter_v3(src_path, cam_rename, cam_rate_col, needed_cols, num_episodes)

    n_eps = 0
    n_frames = 0
    for payload, full_len in episode_iter:
        wrote = _write_episode(
            new_ds, payload, cam_rename, state_cols, action_cols, global_task_text, decoders,
        )
        n_eps += 1
        n_frames += wrote
        print(f"[w{worker_id}] ep {payload.episode_id}: {wrote} frames (orig {full_len})", flush=True)

    new_ds.finalize()
    return output_root, n_eps, n_frames


def _count_v21_rating_passing(src: Path, min_rating: int) -> int:
    """Count v2.1 episodes whose annotation_rating >= min_rating."""
    n = 0
    for mp in sorted((src / "meta").glob("episode_*_metadata.json")):
        rating = (json.loads(mp.read_text()).get("annotation_rating") or 0)
        if rating >= min_rating:
            n += 1
    return n


# ── main flow ───────────────────────────────────────────────────────────────


def convert(
    src: Path,
    dst_repo_id: str,
    num_episodes: int,
    include_ft: bool,
    overwrite: bool,
    min_rating: int,
    skip_episodes: int = 0,
    num_workers: int = 1,
) -> None:
    src_info = json.loads((src / "meta" / "info.json").read_text())
    src_features = dict(src_info["features"])
    codebase_version = src_info.get("codebase_version", "v2.1")

    # v2.1 info.json doesn't enumerate proprio/action features — infer them from
    # a sample parquet's per-element columns and merge in.
    sample_parquet = next((src / "data").glob("chunk-*/*.parquet"))
    inferred = infer_features_from_parquet(sample_parquet)
    for base, ft in inferred.items():
        src_features.setdefault(base, ft)

    cam_rename = detect_cameras(src_features)
    arms = detect_arms(src_features)
    state_cols, state_names = state_recipe(src_features, arms, include_ft)
    action_cols, action_names = action_columns(src_features)

    first_cam_src = next(iter(cam_rename))
    cam_rate_col = f"frame_index.{first_cam_src.removeprefix('observation.')}"
    cam_fps = src_features[first_cam_src].get("fps")
    if cam_fps is None:
        raise ValueError(f"Source camera feature {first_cam_src} missing fps.")

    print(f"[convert] src: {src}")
    print(f"[convert] codebase_version: {codebase_version}")
    print(f"[convert] arms detected: {arms}")
    print(f"[convert] cameras: {cam_rename}")
    print(f"[convert] state dim: {len(state_cols)}{' (includes F/T)' if include_ft else ''}")
    print(f"[convert] action dim: {len(action_cols)}")
    print(f"[convert] target fps: {cam_fps}")
    if codebase_version.startswith("v2"):
        print(f"[convert] min annotation_rating: {min_rating}")

    dst_root = LEROBOT_CACHE / dst_repo_id
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst_root} already exists. Pass --overwrite or pick a different --dst-repo-id."
            )
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)

    # Probe video shape: v2.1 info.json has empty shape, so we read the first
    # frame from a real mp4. For v3, the info.json has it but probing still works.
    decoders = VideoDecoderCache()
    if codebase_version.startswith("v2"):
        sample_meta = next((src / "meta").glob("episode_*_metadata.json"))
        sample_id = _v21_extract_id(sample_meta)
        sample_video = _v21_episode_chunk_dir(
            src, sample_id, "video", cam=first_cam_src.removeprefix("observation.")
        )
    else:
        sample_video = next((src / "videos" / first_cam_src).glob("chunk-*/file-*.mp4"))
    video_shape = decoders.probe_shape(sample_video)
    print(f"[convert] video shape: {video_shape}")

    needed_cols = [cam_rate_col] + state_cols + action_cols

    if codebase_version.startswith("v2"):
        global_task_text = (
            src_info.get("task_config", {}).get("description")
            or src_info.get("task_id")
            or "task"
        )
    elif codebase_version.startswith("v3"):
        if skip_episodes:
            raise ValueError("--skip-episodes is only supported for v2.1 sources.")
        if num_workers > 1:
            raise ValueError("--num-workers > 1 is only supported for v2.1 sources.")
        global_task_text = (
            src_info.get("task_config", {}).get("description")
            or src_info.get("task_id")
            or "task"
        )
    else:
        raise ValueError(f"Unsupported codebase_version: {codebase_version}")

    features = build_features_dict(
        cam_rename, len(state_cols), state_names, len(action_cols), action_names, video_shape
    )

    if num_workers <= 1:
        # Single-process path (unchanged from the original implementation).
        if codebase_version.startswith("v2"):
            episode_iter = iter_v21(
                src, cam_rename, cam_rate_col, needed_cols, min_rating, num_episodes,
                skip_episodes=skip_episodes,
            )
        else:
            episode_iter = iter_v3(src, cam_rename, cam_rate_col, needed_cols, num_episodes)

        new_ds = LeRobotDataset.create(
            repo_id=dst_repo_id,
            fps=int(cam_fps),
            features=features,
            robot_type=src_info.get("robot_type"),
            use_videos=True,
        )
        print(f"[convert] writing to: {new_ds.root}")

        for payload, full_len in episode_iter:
            wrote = _write_episode(
                new_ds, payload, cam_rename, state_cols, action_cols, global_task_text, decoders,
            )
            print(f"[convert] ep {payload.episode_id}: wrote {wrote} frames (orig {full_len} @ source rate)")
        new_ds.finalize()
        print(f"[convert] done. dataset at {new_ds.root}")
        print(f"[convert] total frames: {new_ds.meta.total_frames}, episodes: {new_ds.meta.total_episodes}")
        return

    # ── multi-process path ───────────────────────────────────────────────
    # Each worker converts a disjoint episode slice into its own staging
    # dataset, then we use lerobot.datasets.merge_datasets() to combine them.
    # v2.1 only (v3 path is rejected above).
    total_pass = _count_v21_rating_passing(src, min_rating)
    available = total_pass - skip_episodes
    if available <= 0:
        raise ValueError(
            f"No episodes available after --skip-episodes={skip_episodes} "
            f"with --min-rating={min_rating} ({total_pass} rating-passing total)."
        )
    effective_n = available if num_episodes < 0 else min(num_episodes, available)
    per_worker = math.ceil(effective_n / num_workers)

    slices: list[tuple[int, int]] = []
    for i in range(num_workers):
        slice_skip = skip_episodes + i * per_worker
        slice_n = min(per_worker, effective_n - i * per_worker)
        if slice_n <= 0:
            break
        slices.append((slice_skip, slice_n))

    print(f"[convert] parallel: {len(slices)} workers, ~{per_worker} eps each "
          f"(total {effective_n} eps to process)")

    from lerobot.datasets import merge_datasets

    # Use a temp dir on the same filesystem as the final destination so the
    # merge step can rename files cheaply instead of copying across mounts.
    with tempfile.TemporaryDirectory(prefix="convert_pb_", dir=str(dst_root.parent)) as tmpdir:
        worker_roots = [Path(tmpdir) / f"worker_{i}" for i in range(len(slices))]

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(slices)) as ex:
            futures = []
            for i, ((slice_skip, slice_n), worker_root) in enumerate(zip(slices, worker_roots)):
                fut = ex.submit(
                    _worker_convert_slice,
                    i,
                    str(worker_root),
                    str(src),
                    cam_rename,
                    cam_rate_col,
                    needed_cols,
                    state_cols,
                    action_cols,
                    cam_fps,
                    features,
                    src_info.get("robot_type"),
                    global_task_text,
                    min_rating,
                    slice_skip,
                    slice_n,
                    codebase_version,
                )
                futures.append(fut)
            for f in concurrent.futures.as_completed(futures):
                root, n_eps, n_frames = f.result()
                print(f"[convert] worker done: {Path(root).name} → {n_eps} ep, {n_frames} frames")

        worker_dss = [
            LeRobotDataset(repo_id=f"local-staging/worker-{i}", root=root)
            for i, root in enumerate(worker_roots)
        ]
        print(f"[convert] merging {len(worker_dss)} worker datasets → {dst_root}")
        merged = merge_datasets(
            datasets=worker_dss,
            output_repo_id=dst_repo_id,
            output_dir=dst_root,
        )
        print(f"[convert] done. dataset at {merged.root}")
        print(f"[convert] total frames: {merged.meta.total_frames}, episodes: {merged.meta.total_episodes}")


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
        help="Direct path to the source dataset dir (use for v2.1 local datasets).",
    )
    parser.add_argument(
        "--dst-repo-id",
        help="Destination repo id under $HF_LEROBOT_HOME. "
        "Defaults to '<src-repo-id>-lerobot' (with '-ft' suffix if --include-ft).",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Episodes to convert. -1 for all.")
    parser.add_argument(
        "--skip-episodes",
        type=int,
        default=0,
        help="v2.1 only: skip the first N rating-passing episodes before converting any. "
        "Use to carve train/val splits from the same source.",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=5,
        help="v2.1 only: minimum annotation_rating to keep. 5 = clean only, 4 = also include recovery.",
    )
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Multi-process the conversion across N workers. v2.1 only. "
        "Each worker converts a disjoint episode slice into a staging dataset, "
        "then results are merged via lerobot.datasets.merge_datasets. "
        "4-8 is a reasonable starting point on a modern CPU; tune down if you "
        "see ffmpeg encoder contention.",
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

    convert(
        src, dst_repo_id, args.num_episodes, args.include_ft, args.overwrite, args.min_rating,
        skip_episodes=args.skip_episodes,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
