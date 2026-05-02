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
| `pb-pr-domino-select-v1` | 19,555 | 50,000 | ~82 | _not run_ | **80.0%** (40/50) | ~11.7 h |

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

Set `$TASK` once, then run the same three steps. Raw data is assumed at
`/data/dataset/$TASK/` (downloaded via [scripts/fetch_from_s3.sh](fetch_from_s3.sh)
from `s3://physicsbench-data/lerobot_v21/$TASK/`).

```bash
TASK=pb-pr-edge-slide-v1   # or pb-pr-domino-select-v1, etc.
```

### 3.1 Convert pilot100 split (training data)

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

### 3.2 (Optional) Convert val30 split for Tier-1 eval

```bash
/home/ubuntu/lerobot/.venv/bin/python scripts/convert_physicsbench_to_lerobot.py \
  --src-path /data/dataset/$TASK \
  --dst-repo-id manav-robotics/${TASK}-lerobot-val30 \
  --skip-episodes 100 --num-episodes 30 \
  --num-workers 4
```

Skip this if you only want the Tier-2 (sim) number.

### 3.3 Train DP

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

### 3.4 Tier-1 eval — action MSE on val30 (cheap, ~5 min)

Skip if val30 wasn't converted.

```bash
STEP=30000   # whichever checkpoint you want to score
/home/ubuntu/lerobot/.venv/bin/python scripts/eval_dp_action_mse.py \
  --policy-path /data/lerobot/outputs/train/dp_${TASK}_pilot100/checkpoints/last/pretrained_model \
  --val-repo-id manav-robotics/${TASK}-lerobot-val30 \
  --output /data/lerobot/outputs/eval/dp_${TASK}_step${STEP}/action_mse_full.json
```

### 3.5 Tier-2 eval — PhysicsBench rollout (the real number, ~30–60 min)

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
/data/lerobot/outputs/eval/dp_pilot100_step30000/          ← legacy eval dir
├── action_mse_full.json
├── rollout_50ep.json
├── rollout_videos.json                                    ← 5-seed video run
└── videos/                                                ← 5 ego-cam mp4s
```

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
in §3.2 + §3.4.

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
Cheap to check — see [§3.2](#32-optional-convert-val30-split-for-tier-1-eval)
+ [§3.4](#34-tier-1-eval--action-mse-on-val30-cheap-5-min).

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

## 10. Cross-task observations (so far)

With two tasks done it's still too early to draw conclusions, but worth
noting:

- **Episode length matters more than episode count.** Domino-select has
  100 episodes × ~196 frames each (19.5k frames total) and reached 80%
  success. Edge-slide has 100 × ~720 each (72k frames) and reached 34%.
  Same demo count, very different success — the domino dataset gave the
  policy ~6× more passes per frame at 50k steps vs ~13 passes for
  edge-slide at 30k.
- **Failure profile differs by task.** Edge-slide failures include
  fall-off terminations (object went over the table edge); domino-select
  failures are all timeouts (policy approaches but doesn't finish in time).
  Failure mix is a useful diagnostic in itself.
- **Auto-derived camera map saved time.** Adding domino-select required
  zero code changes to the eval adapter once `_derive_cam_map` was in
  place — it picks up whatever the policy config expects.

## 11. Files added / changed during this work

- [scripts/convert_physicsbench_to_lerobot.py](convert_physicsbench_to_lerobot.py) — added `--skip-episodes` (train/val carving) and `--num-workers` (multi-process via `merge_datasets`).
- [scripts/eval_dp_action_mse.py](eval_dp_action_mse.py) — Tier-1 action MSE (new).
- [scripts/eval_dp_in_physicsbench.py](eval_dp_in_physicsbench.py) — Tier-2 PhysicsBench rollout adapter (new). Auto-derives the per-policy camera map from `policy.config.image_features`. Supports `--seeds`, `--videos-dir`, `--video-camera`.
- [scripts/physicsbench_to_lerobot.md](physicsbench_to_lerobot.md) — fleshed out §C (eval plan), removed the wrong `scheduler_decay_steps` flag from the training recipes.
- This file.
