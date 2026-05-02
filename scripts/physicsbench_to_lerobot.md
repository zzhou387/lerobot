# PhysicsBench → LeRobot: Conversion Guide & First-Run Notes

**Scope.** How to convert any PhysicsBench-format HF dataset into a
LeRobot-policy-ready local dataset, and the findings from the first concrete run
on `manav-robotics/pb-pr-edge-slide-v1` (Diffusion Policy training).

The same converter ([scripts/convert_physicsbench_to_lerobot.py](convert_physicsbench_to_lerobot.py))
works for other PhysicsBench tasks (single-arm and bimanual) — see §9.

**Goal of the first run.** Train a Diffusion Policy on
`manav-robotics/pb-pr-edge-slide-v1` to verify the dataset is high enough
quality for downstream policy work. Quick iteration on this laptop (RTX 4070
Laptop, 8 GB), then the real run on a server.

**Status.** ✅ Conversion + training pipeline is working end-to-end. 10-episode
sanity slice converts in ~30 s; `lerobot-train` runs at ~0.23 s/step with all 4
cameras and a 240×320 crop on the laptop, loss decreasing.

---

## 1. What needed fixing before training

The raw HF dataset doesn't match the schema LeRobot's Diffusion Policy expects.
Three concrete blockers:

| Issue | Source dataset | Diffusion expects |
|---|---|---|
| State key | many proprio columns (`robot0_eef_pos`, `robot0_joint_pos`, …) | a single `observation.state` |
| Camera names | `observation.rgb_{ego,exo_left,exo_right,gripper}_cam` | keys starting with `observation.image*` |
| FPS | multi-rate (cameras 25 Hz, proprio 100 Hz) | single rate |

Plus: the dataset includes **privileged sim state** (`object_pos`, `object_quat`,
`object_state`) — these must be excluded from training or the policy learns
to cheat.

The conversion script [scripts/convert_physicsbench_to_lerobot.py](convert_physicsbench_to_lerobot.py)
fixes all of this in one pass — generalized to any PhysicsBench task, not just
edge-slide. It auto-detects cameras, arms, action dim, video shape, and task
description from the source, and supports **both PhysicsBench codebase versions**
(v2.1 and v3.0) via auto-dispatch on `codebase_version` in `info.json`.

---

## 1b. Source layout: v2.1 vs v3.0

The team has standardized on **PhysicsBench v2.1** for training going forward.
The converter auto-detects which version it's looking at via `codebase_version`
in `info.json` and dispatches to the right reader.

### v2.1 layout (recommended; what to use)
```
<src>/
├── meta/
│   ├── info.json                  # only declares cameras + frame_index; proprio/action inferred from parquet
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── episode_<ID>_metadata.json # ⭐ has annotation_rating ∈ {4, 5}
├── data/chunk-XXX/
│   └── episode_<ID>.parquet       # one parquet per episode
└── videos/chunk-XXX/<cam>/
    └── episode_<ID>.mp4           # one mp4 per episode per camera
```

Per-episode metadata JSONs include `annotation_rating`:
- **5** = clean single-shot success (most common)
- **4** = success with recovery (the operator retried mid-episode)

**Recommendation from the data team:** start sanity-checks with rating = 5 only.
Rating-4 episodes are useful for inducing recovery behavior but add noise to
the demo distribution. The script defaults to `--min-rating 5`; pass
`--min-rating 4` to include both.

In our edge-slide v2.1 dataset (823 episodes total): **751 rating-5, 72
rating-4**. All 823 outcomes = "success".

### v3.0 layout (legacy — first run notes use this)
```
<src>/
├── meta/
│   ├── info.json
│   ├── episodes/chunk-XXX/file-YYY.parquet  # all episodes' meta in one parquet
│   └── tasks.parquet
├── data/chunk-XXX/file-YYY.parquet          # multi-episode; episode-meta has from/to indices
└── videos/<cam>/chunk-XXX/file-YYY.mp4      # multi-episode mp4
```

Episode meta gives positional row offsets (`dataset_from_index`,
`dataset_to_index`) into the shared parquet, plus per-camera `chunk_index/file_index`
identifying the shared mp4. Per-camera frame indices in the data parquet
(`frame_index.rgb_*_cam`) are **global** within each shared mp4.

The first-run findings in §2-§5 below were all collected on v3.0 of edge-slide.
The corresponding v2.1 dataset (more episodes, same task) is the one to actually
train on.

---

## 2. Red flags found in the dataset

Things worth confirming with whoever produced the data:

1. **`episode_index` column in data parquets is internally inconsistent with episode
   meta.** The episodes-meta parquet has 411 entries (sequential 0..410). The
   `episode_index` column in the data parquets has only 234 unique values across
   the range 0..241 with gaps, and the same `episode_index` value appears across
   multiple data files. The episode meta uses positional row offsets
   (`dataset_from_index`/`dataset_to_index`) which work correctly — that's what
   `LeRobotDataset` and our conversion script use. **It just means the
   `episode_index` *column* in the data is legacy / pre-filtering and shouldn't
   be trusted.**

2. **`task_score` is binary (0/1), not the human 1-5 rating.** Every episode's
   final `task_score` is 1.0. The 1-5 quality scoring done outside the dataset
   isn't preserved in the data, so we can't filter further (e.g. drop "score 4
   recovery" episodes).

3. **Some episodes hit the `max_episode_steps=10000` cap.** Episode meta shows
   max length = 10433 frames (over the cap). Worth confirming none of the kept
   episodes were truncated mid-task — those would teach the policy non-terminal
   plateau behavior.

4. **Domain randomization is partial.** Object color, size, friction, and light
   *position* are randomized at collection time. **Light color and table color
   are OFF.** Fine for sim eval; needed for sim2real.

---

## 3. Dataset conversion choices

### `observation.state` (9-dim)

```
[ eef_pos (3) | eef_quat (4) | gripper_qpos (2) ]
```

Picked because the action is `cartesian_delta` in **base frame** (per
[`pb-pr-edge-slide-v1.yaml`](../PhysicsBench-main/configs/tasks/prehensile/pb-pr-edge-slide-v1.yaml)):
matching the action's reference frame is the standard recipe for Diffusion Policy
on Panda-style robots. Equivalent to copying the existing `observation.ee_poses`
column directly into `observation.state`.

**Excluded from state** (deliberate):
- `joint_vel` — adds noise, DP doesn't need it for this task
- `object_*` — privileged sim ground-truth, leaks the answer
- `tactile`, `force_torque` — kept out for v1 simplicity; **F/T is the obvious
  v2 add** since edge-slide is contact-rich

### Camera renames

| Source key | Destination key |
|---|---|
| `observation.rgb_ego_cam` | `observation.images.ego` |
| `observation.rgb_exo_left_cam` | `observation.images.exo_left` |
| `observation.rgb_exo_right_cam` | `observation.images.exo_right` |
| `observation.rgb_gripper_cam` | `observation.images.gripper` |

All 4 cameras kept at native 480×640.

### FPS subsample 100 → 25 Hz

The cameras are physically rendered at 25 Hz at collection time
(see [`base_task.py:163-169`](../PhysicsBench-main/physicsbench/tasks/base_task.py#L163-L169):
`_camera_render_skip = 4` cached frames between renders). At 100 Hz proprio, 4
consecutive rows reference the same physical camera frame — so subsampling to
25 Hz throws away **nothing**, just removes redundant rows.

---

## 4. PhysicsBench design observations affecting training

After reading [`edge_slide_v1.py`](../PhysicsBench-main/physicsbench/tasks/prehensile/edge_slide_v1.py),
[`base_task.py`](../PhysicsBench-main/physicsbench/tasks/base_task.py),
[`collect_demos.py`](../PhysicsBench-main/scripts/collect_demos.py), and the YAML config:

### Confirms our defaults — keep as-is
- **EE-state recipe is correct.** Action is in base-frame Cartesian delta; matching
  state keeps the policy's mapping simple.
- **25 Hz subsample is lossless.** Camera frames are cached at collection time.
- **Pause-frame skipping already done at collection.** `collect_demos.py:589` skips
  the physics step when `paused=True`. No extra filtering needed.
- **Action is already smoothed.** Low-pass `alpha=0.8` + per-step delta clamp
  `max_delta=0.15` (`collect_demos.py:332-365`). DP-friendly demos.

### Subtle gotchas — non-blocking
- **State/action rotation representations don't match.** Action rotation is 3-dim
  (axis-angle/Euler-style deltas, `keyboard.py:49-54`); state rotation is 4-dim
  quaternion. Robosuite IK pipeline limitation; DP can learn this mapping with
  enough data. Slightly less aggressive loss curve than if we used matching
  representations.
- **Reward is sparse, computed end-of-episode** (`edge_slide_v1.py:73-77, 276-295`):
  `1.0` iff object lifted >0.15 m AND gripper-cube contact. **Don't try to use
  reward as a per-step training signal.**
- **Episodes terminate on success or fall-off.** Every kept episode ends in
  success (you filtered to scores 4-5). Action distribution is biased toward the
  successful endgame motion, which DP will learn strongly. Good for this task.

### For the real training run later
- **Add `force_torque` to `observation.state` (→ 15-dim).** Edge-slide is
  contact-rich; F/T at the wrist is privileged signal already in the dataset.
  Bumps the state dim from 9 → 15. Try as v2 once the baseline works.
- **Tactile (8×8 maps @ 50 Hz) is interesting but heavyweight** — needs a CNN
  encoder plumbed through `processor_diffusion`. Defer to v3.

---

## 5. Sanity check results (this laptop)

### Conversion

```
$ uv run python scripts/convert_physicsbench_to_lerobot.py \
    --src-repo-id manav-robotics/pb-pr-edge-slide-v1 --num-episodes 10
[convert] arms detected: [0]
[convert] cameras: {'observation.rgb_ego_cam': 'observation.images.ego', ...}
[convert] state dim: 9
[convert] action dim: 7
[convert] target fps: 25
[convert] episodes available: 411, processing: 10
[convert] ep 1/10: wrote 1172 frames (orig length 4687 @ source rate)
...
[convert] done. total frames: 8842, episodes: 10
```

Wall time: ~30 s on the laptop. Default destination is
`<src-repo-id>-lerobot` under `$HF_LEROBOT_HOME`.

### Training

```
$ uv run lerobot-train \
    --dataset.repo_id=manav-robotics/pb-pr-edge-slide-v1-lerobot \
    --policy.type=diffusion --policy.device=cuda \
    --policy.crop_shape='[240,320]' --policy.push_to_hub=false \
    --batch_size=4 --steps=10 --eval_freq=0 \
    --output_dir=outputs/train/dp_smoke --job_name=dp_smoke

step:1  loss:1.138  updt_s:2.328  data_s:0.330
step:2  loss:1.305  updt_s:0.234
...
step:10 loss:1.085  updt_s:0.246
```

| | |
|---|---|
| Trainable params | 274 M |
| Effective batch size | 4 |
| Step time (after warmup) | ~0.23 s |
| Cameras | 4, cropped to 240×320 |
| GPU peak | fits in 8 GB |

Loss decreasing as expected. **The training pipeline works end-to-end.**

---

## 6. Next steps

### A. Laptop quick-iter (optional)

If you want to keep iterating here before the server, you can re-run with more
episodes / longer training, but stay conservative on memory:

```bash
# Bigger conversion (e.g. 50 episodes — still finishes in a few minutes)
uv run python scripts/convert_physicsbench_to_lerobot.py \
  --src-repo-id manav-robotics/pb-pr-edge-slide-v1 \
  --num-episodes 50 --overwrite

# Longer run, same shape constraints (4 cams cropped, batch 4)
uv run lerobot-train \
  --dataset.repo_id=manav-robotics/pb-pr-edge-slide-v1-lerobot \
  --policy.type=diffusion --policy.device=cuda \
  --policy.crop_shape='[240,320]' --policy.push_to_hub=false \
  --batch_size=4 --steps=2000 \
  --eval_freq=0 --save_freq=500 --log_freq=20 \
  --wandb.enable=false \
  --output_dir=outputs/train/dp_laptop_iter --job_name=dp_laptop_iter
```

> **Caveats.** With 8 GB VRAM you cannot use the full 480×640 res or batch >4
> with all 4 cameras. If you want one of those, drop a camera (delete from the
> dataset features dict, or pass a custom subset of features to the policy via
> overrides — easier to just drop from the conversion script).

### B. Server full run

On the server (assume 24+ GB GPU):

```bash
# 1. Convert ALL 411 episodes
uv run python scripts/convert_physicsbench_to_lerobot.py \
  --src-repo-id manav-robotics/pb-pr-edge-slide-v1 \
  --num-episodes -1 --overwrite
```

Estimate: ~15–25 min for the full conversion (re-encodes ~196k frames × 4 cams).
The script is single-process; if it's painful, parallelizing across episodes is
a straightforward change.

```bash
# 2. Train at full resolution + sensible batch
uv run lerobot-train \
  --dataset.repo_id=manav-robotics/pb-pr-edge-slide-v1-lerobot \
  --policy.type=diffusion --policy.device=cuda \
  --policy.crop_shape='[480,640]' --policy.push_to_hub=false \
  --batch_size=32 --steps=80000 \
  --eval_freq=0 --save_freq=10000 --log_freq=100 \
  --wandb.enable=true \
  --output_dir=outputs/train/dp_edge_slide_v1 \
  --job_name=dp_edge_slide_v1
```

Reasoning for these numbers (per [`AGENT_GUIDE.md`](../AGENT_GUIDE.md) §7):
- 80k steps ≈ ~13 epochs at batch 32 over ~196k frames — within the recommended
  80k–150k range for single-task DP.
- LR schedule: `DiffusionConfig` uses HF diffusers' cosine scheduler with
  `scheduler_warmup_steps=500`, and decay automatically spans the remaining
  `--steps`. There is **no** `scheduler_decay_steps` flag on
  `DiffusionConfig` (that field exists on other policies like SmolVLA, but
  passing it to DP raises `DecodingError`).
- `eval_freq=0` because **there is no sim env registered for this task** in
  LeRobot — eval needs a separate path (replay-based or running PhysicsBench
  itself with the trained policy).

### C. Eval (no in-training rollout possible)

There's no `--env.type=pb-pr-edge-slide-v1` registered in
[`src/lerobot/envs/`](../src/lerobot/envs/) so `lerobot-eval` can't drive the
sim directly. Two-tier eval below: action-MSE first as a cheap screen,
PhysicsBench-driven rollout second for the real number.

#### Tier 1 — Action-MSE on held-out split (cheap, ~5 min, CPU-OK)

Detects over/under-fit and screens checkpoints before paying for sim rollouts.

```bash
# 1. Convert a held-out 30-episode val slice (next 30 rating-5 episodes after the
#    first 100 we trained on)
uv run python scripts/convert_physicsbench_to_lerobot.py \
  --src-path /home/ubuntu/dataset \
  --dst-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot-val30 \
  --skip-episodes 100 --num-episodes 30

# 2. Run action-MSE on each saved checkpoint
uv run python scripts/eval_dp_action_mse.py \
  --policy-path outputs/train/dp_edge_slide_pilot100/checkpoints/last/pretrained_model \
  --val-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot-val30 \
  --output outputs/eval/dp_pilot100/action_mse.json
```

Reports per-step MSE + per-dim breakdown. Use to pick the best checkpoint
before the rollout step.

#### Tier 2 — PhysicsBench rollout (the real eval, ~30–45 min for 50 eps)

PhysicsBench exposes its own gym-style env via `pb.make(task_id, ...)` and
already has an eval harness ([`physicsbench/eval/harness.py`](../PhysicsBench/physicsbench/eval/harness.py))
that wraps an `act(obs)/reset()` policy. We add a thin LeRobot adapter that
runs episodes directly (skipping the harness so we can pass `config_overrides`
for the FPS subtlety below).

PhysicsBench is vendored under [`PhysicsBench/`](../PhysicsBench/), not
installed into the lerobot venv — its sim deps (`robosuite`, `mujoco`,
`opencv-python`, `lxml`, `flask`, `pyarrow`) need to be installed alongside
lerobot before rollouts will run:

```bash
uv pip install "robosuite>=1.4" "mujoco>=3.0,<3.3" opencv-python lxml flask pyarrow
```

The eval script auto-adds `PhysicsBench/` to `sys.path`, so `physicsbench`
imports without a separate install step.

```bash
uv run python scripts/eval_dp_in_physicsbench.py \
  --policy-path outputs/train/dp_edge_slide_pilot100/checkpoints/last/pretrained_model \
  --task-id pb-pr-edge-slide-v1 \
  --n-episodes 50 --seed-offset 1000 \
  --output outputs/eval/dp_pilot100/results.json
```

Reports `mean_score`, `std_score`, `success_rate (>0.9)`, `mean_length`, plus
per-episode dump. Defaults to `obs_mode="combo"` (proprio+cameras) per
[`PhysicsBench/CLAUDE.md`](../PhysicsBench/CLAUDE.md).

For faster rollouts, the script switches the diffusion scheduler to DDIM with
`num_inference_steps=10` at load time (~10× faster than the DDPM-100 default
at near-equal quality).

#### Subtlety to validate before trusting the rollout number

**FPS mismatch.** Training data is subsampled 100→25 Hz; PhysicsBench's
`edge_slide_v1` defaults to `control_freq=100`. DP outputs an action expecting
25 Hz application. Three options, in order of cleanness:

1. Override `control_freq=25` when calling `pb.make(...)` — closes the gap.
2. Apply DP action at 25 Hz, hold-zero for the other 3/4 ticks at 100 Hz —
   matches what subsampled data implies (assumes deltas are per-step).
3. Apply DP action and repeat 4× — will overshoot, wrong.

The eval script uses option (1) by default. Sanity-check on 5 eps with
options (1) and (2) before committing to the full 50-ep run; if the success
rate is materially different, the FPS recipe matters.

#### Output directory layout

```
outputs/eval/dp_pilot100/
├── action_mse.json          # Tier 1: per-step + per-dim MSE
├── results.json             # Tier 2: aggregated + per-episode rollout metrics
└── videos/                  # if --save-videos: rendered ego cam per episode
```

---

## 7. Open questions for the data team

1. Why does the data parquet `episode_index` column not match the episode-meta
   indices? (Cosmetic, but downstream tools naively reading the column will
   group frames wrong.)
2. Did any kept episodes hit the 10000-step truncation cap? Quickest check:
   `length == 10000` or `length > 10000` in the episode-meta parquet.
3. Are there separate datasets for the bimanual tasks? The user mentioned
   "bimanual task and the edge slide" — this dataset is single-arm only
   (`action_dim=7`, `robot_type=Panda`). The bimanual ones would be `pb-hetbi-*`
   or `pb-hobi-*` per [`PhysicsBench README.md`](../PhysicsBench-main/README.md).
4. For sim2real later: enable lighting **color** + table color randomization in
   the task YAML.

---

## 8. Files we created / modified

- [scripts/convert_physicsbench_to_lerobot.py](convert_physicsbench_to_lerobot.py) —
  generalized PhysicsBench → LeRobot converter. Auto-detects cameras, arms,
  action dim, video shape, and task description from the source. Reusable for
  other tasks (single-arm and bimanual) and both codebase versions (v2.1, v3.0).
  Flags:
    - `--src-repo-id <id>` (HF Hub cache lookup) **or** `--src-path <dir>` (local v2.1)
    - `--dst-repo-id <id>` (default: `<src>-lerobot` or `<src>-lerobot-ft`)
    - `--num-episodes N` (default 10; `-1` for all)
    - `--min-rating N` (v2.1 only; default 5 = clean only, 4 = include recovery)
    - `--include-ft` to append `observation.force_torque` to `observation.state`
    - `--overwrite` to delete an existing destination
- [scripts/physicsbench_to_lerobot.md](physicsbench_to_lerobot.md) — this guide / first-run notes.
- `~/.cache/huggingface/lerobot/<src-repo-id>-lerobot/` — converted local
  dataset (not pushed to Hub).
- `outputs/train/dp_smoke/` — sanity-test checkpoint.

## 9. Reusing the converter on other PhysicsBench tasks

The converter is task-agnostic and version-agnostic. Examples:

```bash
# v2.1 local dataset, only clean (rating-5) episodes
uv run python scripts/convert_physicsbench_to_lerobot.py \
  --src-path ~/manav/dataset/pb-pr-edge-slide-v1 \
  --dst-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot \
  --num-episodes -1

# v2.1, include rating-4 (recovery) too
uv run python scripts/convert_physicsbench_to_lerobot.py \
  --src-path ~/manav/dataset/pb-pr-edge-slide-v1 \
  --dst-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot-r4 \
  --num-episodes -1 --min-rating 4

# v2.1 bimanual task
uv run python scripts/convert_physicsbench_to_lerobot.py \
  --src-path ~/manav/dataset/pb-hetbi-bowl-table-v1 \
  --dst-repo-id manav-robotics/pb-hetbi-bowl-table-v1-lerobot \
  --num-episodes 10

# v3.0 from HF Hub cache (legacy)
hf download --repo-type=dataset manav-robotics/pb-pr-edge-slide-v1
uv run python scripts/convert_physicsbench_to_lerobot.py \
  --src-repo-id manav-robotics/pb-pr-edge-slide-v1 \
  --num-episodes 10
```

For a bimanual task it auto-detects both arms and concatenates per-arm
`[eef_pos, eef_quat, gripper_qpos]` into `observation.state` (typically 18-dim
for two finger-grippers; less for suction since suction grippers have 1-dof
qpos). State columns are named `r0_eef_x`, `r1_eef_x`, etc. when more than one
arm is present.

What's still single-recipe (would need code changes for other shapes):

- The state recipe is always EE-pose + gripper qpos per arm. Using joint-state
  instead would need a small edit in `state_recipe()`.
- The script assumes PhysicsBench naming (`observation.robotN_*`,
  `observation.rgb_*_cam`). Datasets with different naming conventions need
  the regexes in `detect_cameras` / `detect_arms` extended.
- Privileged sim-state keys are dropped via the whitelist (we only KEEP what's
  built), so any new privileged key is auto-excluded.
- Only PhysicsBench codebase versions v2.1 and v3.0 are supported. New layouts
  would need a new `iter_<version>` reader function and a dispatch branch in
  `convert()`.
