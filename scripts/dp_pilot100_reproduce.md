# DP pilot-100 reproducibility report

End-to-end recipe for training Diffusion Policy on a single PhysicsBench
task in the **100-episode pilot** configuration, plus a per-task results
log. Companion to [physicsbench_to_lerobot.md](physicsbench_to_lerobot.md)
(general conversion guide).

Pattern: pick a `$TASK`, run the same three steps (convert → train → eval),
record numbers in [§9 Results](#9-results-per-task). The recipe is unchanged
across tasks; only the per-task numbers and dataset shapes differ.

## Summary table

| Task | Train frames | Steps | Epochs | Tier-1 overall MSE | **Tier-2 success rate (50 ep)** | Train wall |
|---|---:|---:|---:|---:|---:|---:|
| `pb-pr-edge-slide-v1` | 72,367 | 30,000 | ~13 | 0.0215 | **34.0%** (17/50) | ~8.6 h |
| `pb-pr-edge-slide-v1` (cont'd to 50k) | 72,367 | 50,000 | ~22 | _not run_ | 28.0% (14/50) — **worse** ([overfit](#911-step-50k-overfit-followup)) | ~13.4 h |
| `pb-pr-domino-select-v1` | 19,555 | 50,000 | ~82 | _not run_ | **80.0%** (40/50) | ~11.7 h |
| `pb-pr-domino-single-v1` | 16,245 | 50,000 | ~98 | _not run_ | **70.0%** (35/50) | ~11.7 h |

(Add new rows as more tasks come online.)

## 1. Hardware / OS assumed

- 1× L40S (46 GB) or comparable. Smoke fits in 8 GB.
- Linux x86_64 with CUDA. PhysicsBench / robosuite / mujoco only support 64-bit Linux/macOS.
- ≥ 60 GB free disk for converted datasets + checkpoints + eval. We use
  `/data` as a separate disk mount (root was tight, see [§2.1](#21-disk-layout)).

## 2. Setup (one-time)

### 2.1 Disk layout

Caches and outputs all live on `/data` (a 412 GB mount). LeRobot honors
two env vars ([utils/constants.py:67](../src/lerobot/utils/constants.py#L67)):

```bash
mkdir -p /data/lerobot/cache/huggingface /data/lerobot/outputs

cat >> ~/.bashrc <<'EOF'
export HF_LEROBOT_HOME=/data/lerobot/cache/huggingface/lerobot
export HF_HOME=/data/lerobot/cache/huggingface
EOF
source ~/.bashrc
```

Layout that results:

```
/data/lerobot/
├── cache/huggingface/
│   ├── lerobot/manav-robotics/      # converted LeRobot datasets
│   └── datasets/                    # HF datasets cache
└── outputs/
    ├── train/dp_<TASK>_pilot100/    # checkpoints + wandb (per task)
    └── eval/dp_<TASK>_step<N>/      # action_mse + rollout results + videos
```

### 2.2 LeRobot install (uv, Python 3.12)

Per the official guide ([huggingface.co/docs/lerobot/installation](https://huggingface.co/docs/lerobot/installation)),
PyTorch ≥ 2.10 path:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~/lerobot
sudo apt install -y ffmpeg          # torchcodec ≥ 0.10 links to system ffmpeg
uv python install 3.12
uv venv --python 3.12

# Both extras matter: dataset (for converters), training (for the trainer)
uv pip install -e ".[dataset,training]"
```

Verify:

```bash
/home/ubuntu/lerobot/.venv/bin/python -c \
  "import pandas, torchcodec, datasets, accelerate, wandb, lerobot; \
   print('OK', lerobot.__file__)"
```

### 2.3 PhysicsBench install (for Tier-2 eval only)

PhysicsBench is vendored at `~/lerobot/PhysicsBench/`. Add its sim deps to
the same venv:

```bash
uv pip install "robosuite>=1.4" "mujoco>=3.0,<3.3" opencv-python lxml flask pyarrow
```

The eval script auto-adds `PhysicsBench/` to `sys.path`, so no separate
install of the package itself.

## 3. Per-task recipe

Set `$TASK` once, then run the steps below in order. Raw data ends up at
`/data/dataset/$TASK/`; converted data, checkpoints, and eval results all
land under `/data/lerobot/...` per [§2.1](#21-disk-layout).

```bash
TASK=pb-pr-edge-slide-v1   # or pb-pr-domino-select-v1, etc.
```

### 3.1 Fetch raw data from S3

Source bucket / region in [scripts/config.conf](config.conf) —
`physicsbench-data` / `ap-south-1`. Layout convention on S3 is
`lerobot_v21/<task_id>/{meta,data,videos}/`.

```bash
./scripts/fetch_from_s3.sh lerobot_v21/$TASK /data/dataset/$TASK
```

Uses `aws s3 sync` so it's idempotent (re-running skips files that already
match by size + timestamp). Pass `--dry-run` first if you're unsure how big
the pull will be:

```bash
./scripts/fetch_from_s3.sh --dry-run lerobot_v21/$TASK /data/dataset/$TASK
```

Each task is typically ~5–15 GB. The script aborts up front if disk space is
insufficient.

### 3.2 Convert pilot100 split (training data)

```bash
/home/ubuntu/lerobot/.venv/bin/python scripts/convert_physicsbench_to_lerobot.py \
  --src-path /data/dataset/$TASK \
  --dst-repo-id manav-robotics/${TASK}-lerobot-pilot100 \
  --num-episodes 100 \
  --num-workers 4
```

Default `--min-rating 5` keeps only clean single-shot demos. With 4 workers
this lands in ~25 min instead of ~90 min single-process. Output goes to
`$HF_LEROBOT_HOME/manav-robotics/${TASK}-lerobot-pilot100/`.

### 3.3 (Optional) Convert val30 split for Tier-1 eval

```bash
/home/ubuntu/lerobot/.venv/bin/python scripts/convert_physicsbench_to_lerobot.py \
  --src-path /data/dataset/$TASK \
  --dst-repo-id manav-robotics/${TASK}-lerobot-val30 \
  --skip-episodes 100 --num-episodes 30 \
  --num-workers 4
```

Skip this if you only want the Tier-2 (sim) number.

### 3.4 Train DP

```bash
wandb login   # one-time, paste API key from https://wandb.ai/authorize

/home/ubuntu/lerobot/.venv/bin/lerobot-train \
  --dataset.repo_id=manav-robotics/${TASK}-lerobot-pilot100 \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.crop_shape='[480,640]' \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=30000 \
  --eval_freq=0 --save_freq=5000 --log_freq=50 \
  --wandb.enable=true --wandb.project=physicsbench \
  --wandb.notes="DP pilot, ${TASK}, 100 rating-5 episodes, full-res" \
  --output_dir=/data/lerobot/outputs/train/dp_${TASK}_pilot100 \
  --job_name=dp_${TASK}_pilot100
```

Adjust `--steps` based on the dataset size if needed. ~13 epochs is the
typical DP sweet spot; rough rule: `steps ≈ 13 × frames / batch_size`. For
a 70k-frame dataset that's 30k steps; for a 20k-frame dataset, 8k steps
gives the same epoch count (but in practice we've trained domino-select
much longer — see [§9.2](#92-pb-pr-domino-select-v1)).

#### Important gotchas

- **Don't pass `--policy.scheduler_decay_steps`** — `DiffusionConfig`
  doesn't have that field. DP uses HF diffusers' cosine scheduler with
  `scheduler_warmup_steps=500` and decay automatically spans `--steps`.
- **Resume after a crash**: the `last` symlink correctly points at the
  most recent **complete** checkpoint, even if a later partial-write
  exists. To resume:
  ```bash
  /home/ubuntu/lerobot/.venv/bin/lerobot-train \
    --config_path=/data/lerobot/outputs/train/dp_${TASK}_pilot100/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --output_dir=/data/lerobot/outputs/train/dp_${TASK}_pilot100
  ```
  `--config_path` must point at a real local file. `HFValidationError:
  Repo id must be in the form…` means the path is wrong.

### 3.5 Tier-1 eval — action MSE on val30 (cheap, ~5 min)

Skip if val30 wasn't converted.

```bash
STEP=30000   # whichever checkpoint you want to score
/home/ubuntu/lerobot/.venv/bin/python scripts/eval_dp_action_mse.py \
  --policy-path /data/lerobot/outputs/train/dp_${TASK}_pilot100/checkpoints/last/pretrained_model \
  --val-repo-id manav-robotics/${TASK}-lerobot-val30 \
  --output /data/lerobot/outputs/eval/dp_${TASK}_step${STEP}/action_mse_full.json
```

### 3.6 Tier-2 eval — PhysicsBench rollout (the real number, ~30–60 min)

```bash
export PYTHONUNBUFFERED=1   # otherwise stdout buffers when piped

/home/ubuntu/lerobot/.venv/bin/python -u scripts/eval_dp_in_physicsbench.py \
  --policy-path /data/lerobot/outputs/train/dp_${TASK}_pilot100/checkpoints/last/pretrained_model \
  --task-id $TASK \
  --n-episodes 50 --seed-offset 1000 \
  --max-episode-steps 1000 \
  --videos-dir /data/lerobot/outputs/eval/dp_${TASK}_step${STEP}/videos \
  --output /data/lerobot/outputs/eval/dp_${TASK}_step${STEP}/rollout_50ep.json
```

The adapter auto-derives the camera map from the policy config, so no
per-task code changes are needed when the camera layout differs (e.g.
edge-slide has `ego/exo_left/exo_right/gripper` while domino-select has
`exo_front/exo_left/exo_right/gripper`). Default `--video-camera` is
`rgb_ego_cam` — override with one of the actually-present cameras for
tasks that don't have an ego cam.

`--control-freq` defaults to **25** to match the dataset's 25 Hz subsample
(PhysicsBench tasks default to 100 Hz; running the env at 100 Hz with a
policy trained on 25 Hz subsampled deltas overshoots).

### 3.7 (Optional) Push checkpoint to S3

For backup or sharing across machines. Push only the inference-ready
`pretrained_model/` directory (~3 GB) — skipping `training_state/` halves
the upload size, and you don't need it unless you intend to resume training:

```bash
STEP=30000   # whichever checkpoint to back up; or use 'last'

./scripts/push_to_s3.sh \
  /data/lerobot/outputs/train/dp_${TASK}_pilot100/checkpoints/${STEP}/pretrained_model \
  manav/checkpoints/dp_${TASK}_pilot100_step${STEP}
```

Uses `aws s3 sync` (add/update only — won't delete remote files).

To push the full run dir (checkpoints + wandb logs, ~13 GB):

```bash
./scripts/push_to_s3.sh \
  /data/lerobot/outputs/train/dp_${TASK}_pilot100 \
  manav/runs/dp_${TASK}_pilot100
```

`--dry-run` works the same as for fetch; `--delete` is opt-in for one-way
mirroring (rarely what you want).

To pull the checkpoint back on another machine:

```bash
./scripts/fetch_from_s3.sh \
  manav/checkpoints/dp_${TASK}_pilot100_step${STEP} \
  /data/lerobot/outputs/train/dp_${TASK}_pilot100/checkpoints/${STEP}/pretrained_model
```

## 4–8. Caveats and global notes

(Renumbered from the original write-up; same content as before.)

1. **Action-MSE is a noisy proxy for chunked policies.** Single-step MSE
   missed a ~34%-success policy on edge-slide and reported "mean-predictor."
   DP's strength is the smoothed action chunk it produces over the receding
   horizon, which single-step MSE doesn't measure. Treat MSE as a directional
   signal only, not a pass/fail bar.
2. **DP is non-deterministic across reruns on the same env seed.** Same
   `env.reset(seed=k)` + same checkpoint → different success outcome on
   reruns, because policy noise is sampled fresh each `select_action()`
   call ([modeling_diffusion.py:243-253](../src/lerobot/policies/diffusion/modeling_diffusion.py#L243-L253)).
   Adds ~5–8% extra noise to a 50-ep success-rate measurement on top of
   binomial sampling noise.
3. **FPS recipe matters.** Training data is 25 Hz subsampled, env defaults
   to 100 Hz. We pass `config_overrides={"control_freq": 25}`. Running the
   env at 100 Hz with a policy trained on 25 Hz subsampled deltas overshoots.
4. **No sim env registered with `lerobot-eval`.** That's why we wrote a
   separate adapter ([scripts/eval_dp_in_physicsbench.py](eval_dp_in_physicsbench.py))
   instead of using `lerobot-eval --env.type=...`.
5. **PhysicsBench `run_benchmark` works** but doesn't accept `config_overrides`,
   which is why the adapter doesn't delegate to it. Same metrics
   (`mean_score`, `success_rate (>0.9)`, `mean_length`), same scoring
   contract.
6. **Resume requires the original output_dir layout.** If you `mv` the run
   directory, keep the `dp_<task>_pilot100/checkpoints/<step>/...` path
   intact relative to the new root. The relative `last → <step>` symlink
   survives `mv` as long as the parent dir moves as a whole.

## 9. Results (per task)

### 9.1 `pb-pr-edge-slide-v1`

**Dataset (pilot100):**

| | |
|---|---|
| Episodes / frames | 100 / 72,367 (avg ~720 frames/ep ≈ 29 sim-seconds) |
| Source | 823 raw episodes from S3, top-100 by rating |
| Cameras | `ego, exo_left, exo_right, gripper` (480×640) |
| State / action | 9-dim / 7-dim |
| Conversion wall time | ~1 h 35 min (single-process; the multi-worker option came later) |

**Training:**

| | |
|---|---|
| Final step | 30,000 (≈ 13 epochs) |
| Final train loss | 0.0184 |
| Total wall time | ~8.6 h on L40S (across original run + one resume) |
| Checkpoints saved | 15000, 20000, 25000, 30000 (`last` → 30000) |
| Wandb run id | `149mp100` (project `physicsbench`) |
| Disk | ~12 GB (4 × ~3 GB checkpoints + wandb logs) |

**Tier-1 (action MSE on val30, 22,420 frames):**

| Dim | Meaning | RMSE | GT std | **NRMSE** |
|---|---|---|---|---|
| 0 | dx | 0.141 | 0.116 | **1.22** |
| 1 | dy | 0.144 | 0.154 | **0.93** |
| 2 | dz | 0.104 | 0.103 | **1.01** |
| 3 | droll | 0.048 | 0.060 | **0.80** |
| 4 | dpitch | 0.0013 | 0.0011 | (std≈0, ignore) |
| 5 | dyaw | 0.0008 | ~0 | (std≈0, ignore) |
| 6 | gripper | 0.311 | 0.594 | **0.52** |
| **overall** | | | | MSE **0.0215** |

NRMSE ≈ 1.0 on translation deltas → looks mean-predictor level. Misleading
— see caveat 1.

**Tier-2 (PhysicsBench rollout, 50 ep, step 30000):**

| | |
|---|---|
| **Success rate** | **34.0%** (17 / 50), 95% CI ≈ 22–48% |
| Mean score | 0.340 ± 0.474 |
| Mean length | 852 / 1000 step cap |
| Wall time | 61 min (73 s/ep) |
| Failure mode | mostly 1000-step truncations; a few short-length fall-off terminations |

**Artifacts:**

```
/data/lerobot/cache/huggingface/lerobot/manav-robotics/
├── pb-pr-edge-slide-v1-lerobot-pilot100/
└── pb-pr-edge-slide-v1-lerobot-val30/
/data/lerobot/outputs/train/dp_edge_slide_pilot100/        ← legacy job_name
/data/lerobot/outputs/eval/dp_pilot100_step30000/          ← legacy eval dir (step-30k)
├── action_mse_full.json
├── rollout_50ep.json
├── rollout_videos.json                                    ← 5-seed video run
└── videos/                                                ← 5 ego-cam mp4s
/data/lerobot/outputs/eval/dp_pb-pr-edge-slide-v1_step50000/  ← step-50k follow-up
├── rollout_50ep.json
└── videos/                                                ← 50 ego-cam mp4s
```

#### 9.1.1 step-50k overfit follow-up

After the original 30k run, training was continued to **step 50,000** (≈22
epochs over the 72k-frame pilot). Tier-2 was re-run on the same seeds
(1000–1049) with the same `--max-episode-steps=1000`:

| | step 30k (13 epochs) | **step 50k (22 epochs)** |
|---|---:|---:|
| Final train loss | 0.0184 | 0.0167 |
| **Success rate (50 ep)** | **34.0%** (17/50) | **28.0%** (14/50) |
| 95% CI | 22–48% | 17–41% |
| Mean length | 852 | 911 / 1000 |
| Failure mode | mixed (truncations + fall-offs) | **92% (33/36) hit cap** |
| Wall time | 61 min | 67 min |

**Read.** Train loss kept dropping but rollout success dropped 6 percentage
points. CIs overlap so it's not stat-sig as a single comparison, but the
direction matches the DP overfit prediction (10–15 epochs is the typical
sweet spot; 22 is past it for a 72k-frame dataset). Failure profile also
shifted toward "trying but timing out" — consistent with the policy
memorizing a slightly off trajectory and replaying it stubbornly.

**Action**: keep using step-30000 as the operational edge-slide checkpoint.
The 50k checkpoints aren't worth backing up.

### 9.2 `pb-pr-domino-select-v1`

**Dataset (pilot100):**

| | |
|---|---|
| Episodes / frames | 100 / 19,555 (avg ~196 frames/ep ≈ 7.8 sim-seconds — short task) |
| Source | raw v2.1 from S3 at `/data/dataset/pb-pr-domino-select-v1/` |
| Cameras | `exo_front, exo_left, exo_right, gripper` (480×640) — note: no `ego` cam |
| State / action | 9-dim / 7-dim |

**Training:**

| | |
|---|---|
| Final step | 50,000 (≈ **82 epochs** — well above DP's typical 10–15 sweet spot) |
| Final train loss | 0.0126 |
| Total wall time | ~11.7 h on L40S |
| Checkpoints saved | 5000, 10000, …, 50000 (`last` → 50000) |
| Wandb run id | `8hmaoc87` (project `physicsbench`) |

**Tier-1**: not run (val30 not converted). Could be added with the recipe
in §3.3 + §3.5.

**Tier-2 (PhysicsBench rollout, 50 ep, step 50000):**

| | |
|---|---|
| **Success rate** | **80.0%** (40 / 50), 95% CI ≈ 67–89% |
| Mean score | 0.800 ± 0.400 |
| Mean length | 428 / 1000 step cap |
| Wall time | 36 min (43.7 s/ep) |
| Failure mode | all 10 failures hit the 1000-step cap as truncations — no fall-off / fail-state terminations |
| Successful-episode mean length | 286 steps (~11.4 s) |

**Caveat worth flagging**: 82 epochs is well into overfit territory for DP.
Without the val30 numbers, we can't say whether this 80% reflects
generalization or memorization of the pilot demos. The fact that all 10
failures were timeouts (not fail states) is consistent with both stories.
Cheap to check — see [§3.3](#33-optional-convert-val30-split-for-tier-1-eval)
+ [§3.5](#35-tier-1-eval--action-mse-on-val30-cheap-5-min).

**Video capture**: this run saved videos for **all 50 episodes** to
`/data/lerobot/outputs/eval/dp_pb-pr-domino-select-v1_step50000/videos/`
(16 MB total at `rgb_exo_front_cam`). Filenames encode outcome —
to find the failures: `ls .../videos/ | grep _fail_`.

**Artifacts:**

```
/data/lerobot/cache/huggingface/lerobot/manav-robotics/
└── pb-pr-domino-select-v1-lerobot-pilot100/
/data/lerobot/outputs/train/dp_pb-pr-domino-select-v1_pilot100/
└── checkpoints/{5000,10000,...,50000,last}/
/data/lerobot/outputs/eval/dp_pb-pr-domino-select-v1_step50000/
├── rollout_50ep.json
└── videos/                                                ← 50 exo_front_cam mp4s
```

### 9.3 `pb-pr-domino-single-v1`

**Dataset (pilot100):**

| | |
|---|---|
| Episodes / frames | 100 / 16,245 (avg ~162 frames/ep ≈ 6.5 sim-seconds — even shorter than domino-select) |
| Source | raw v2.1 from S3 at `/data/dataset/pb-pr-domino-single-v1/` |
| Cameras | `exo_front, exo_left, exo_right, gripper` (480×640) — same set as domino-select, no `ego` cam |
| State / action | 9-dim / 7-dim |

**Training:**

| | |
|---|---|
| Final step | 50,000 (≈ **98 epochs** — even further into overfit territory than domino-select) |
| Final train loss | 0.01027 |
| Total wall time | ~11.7 h on L40S |
| Checkpoints saved | 5000, 10000, …, 50000 (`last` → 50000) |
| Wandb run id | `yt97uqbs` (project `physicsbench`) |

**Tier-1**: not run.

**Tier-2 (PhysicsBench rollout, 50 ep, step 50000):**

| | |
|---|---|
| **Success rate** | **70.0%** (35 / 50), 95% CI ≈ 56–81% |
| Mean score | 0.774 ± 0.377 |
| Mean length | 420.7 / 1000 step cap |
| Wall time | 35 min (41.5 s/ep) |
| Successful-episode mean length | 250 steps (~10 s) |
| Failure-episode mean length | 820 steps |
| Failure breakdown | **10 timeouts** (hit 1000-step cap) + **5 mid-episode terminations** (`terminated=True`, score≤0.9) |

**Note on `mean_score` vs `success_rate`**: 0.774 vs 0.700 — they don't align
because some non-success episodes have non-zero partial scores (~0.25 on
average across the 15 failures). Domino-single's task scoring isn't strictly
binary unlike edge-slide's; partial credit accumulates for partially-completed
sub-goals.

**Failure profile differs from domino-select**: domino-select had all 10
failures hit the 1000-step cap (no fail-states); domino-single had 5/15
mid-episode terminations. Suggests domino-single has a stricter termination
condition (e.g. wrong domino touched → episode ends, vs domino-select
where the policy just keeps trying).

**Video capture**: this run saved videos for **all 50 episodes** to
`/data/lerobot/outputs/eval/dp_pb-pr-domino-single-v1_step50000/videos/`
(17 MB total at `rgb_exo_front_cam`). To find the failures with
mid-episode termination (the more diagnostic ones):

```bash
ls /data/lerobot/outputs/eval/dp_pb-pr-domino-single-v1_step50000/videos/ | grep _fail_ | grep -v _len1000
```

**Artifacts:**

```
/data/lerobot/cache/huggingface/lerobot/manav-robotics/
└── pb-pr-domino-single-v1-lerobot-pilot100/
/data/lerobot/outputs/train/dp_pb-pr-domino-single-v1_pilot100/
└── checkpoints/{5000,10000,...,50000,last}/
/data/lerobot/outputs/eval/dp_pb-pr-domino-single-v1_step50000/
├── rollout_50ep.json
└── videos/                                                ← 50 exo_front_cam mp4s
```

## 10. Cross-task observations (so far)

With three tasks done it's still early, but two patterns are starting to
hold up:

- **Demo *coverage* (frames) matters more than demo *count* (episodes).**
  All three tasks used 100-episode pilots, but their frame counts vary
  ~4.5×, and that — not the demo count — tracks the success rate:

  | Task | Frames | Epochs at last step | Tier-2 success |
  |---|---:|---:|---:|
  | edge-slide | 72k | 13 | 34% |
  | domino-select | 19.5k | 82 | 80% |
  | domino-single | 16.2k | 98 | 70% |

  The two short-episode domino tasks both massively over-train (82–98
  epochs, well beyond DP's usual 10–15 sweet spot) and still produce
  strong rollouts. Edge-slide trained ~6× *less* per frame and produced
  the worst rollout. Hard to disentangle "task complexity" from "how
  many epochs" without holding one fixed.

  **Update from the step-50k follow-up on edge-slide** (see [§9.1.1](#911-step-50k-overfit-follow-up)):
  continuing edge-slide training to step 50,000 (~22 epochs) **dropped**
  success rate from 34% → 28%. So "more training" alone doesn't close
  the gap on edge-slide — at this dataset size DP plateaus and starts to
  overfit. The cross-task gap is more likely task complexity (contact-rich
  edge-sliding-then-grasping is genuinely harder than domino selection)
  than under-training of edge-slide. The likely lever is **more demo
  coverage** (full 751-episode dataset), not more steps.
- **Failure profile differs by task and is a useful diagnostic.**
  Different tasks have different failure modes:
  - Edge-slide: mix of 1000-step timeouts and short-length fall-off
    terminations (object went over the table edge).
  - Domino-select: 100% timeouts — policy doesn't violate state, just
    runs out of time.
  - Domino-single: 67% timeouts + 33% mid-episode terminations
    (probably wrong-domino-touched). More sensitive termination
    condition than domino-select.
  Distinguishing "the policy never destabilizes but doesn't finish" vs
  "the policy actively breaks the task" is informative — the former
  responds well to more steps / more demos; the latter often needs
  task-recipe tweaks.
- **Auto-derived camera map saved time.** Both domino tasks required
  zero code changes to the eval adapter once `_derive_cam_map` was in
  place — it picks up whatever the policy config expects, even though
  edge-slide uses `ego` and the domino tasks use `exo_front`.
- **Domino tasks have non-binary task scores.** edge-slide reports
  `score ∈ {0, 1}`; domino-single's failed episodes carry partial
  credit (~0.25 mean for the 15 failures), so `mean_score` and
  `success_rate` diverge slightly. Watch for both numbers when
  interpreting future tasks.

## 11. Files added / changed during this work

- [scripts/convert_physicsbench_to_lerobot.py](convert_physicsbench_to_lerobot.py) — added `--skip-episodes` (train/val carving) and `--num-workers` (multi-process via `merge_datasets`).
- [scripts/eval_dp_action_mse.py](eval_dp_action_mse.py) — Tier-1 action MSE (new).
- [scripts/eval_dp_in_physicsbench.py](eval_dp_in_physicsbench.py) — Tier-2 PhysicsBench rollout adapter (new). Auto-derives the per-policy camera map from `policy.config.image_features`. Supports `--seeds`, `--videos-dir`, `--video-camera`.
- [scripts/physicsbench_to_lerobot.md](physicsbench_to_lerobot.md) — fleshed out §C (eval plan), removed the wrong `scheduler_decay_steps` flag from the training recipes.
- This file.
