# DP pilot-100 on edge-slide v2.1 — reproducibility report

End-to-end record of the first concrete training + evaluation run on
`pb-pr-edge-slide-v1`: 100 rating-5 episodes through Diffusion Policy, then a
50-episode rollout in PhysicsBench. Every command needed to reproduce the
result is here. Companion to [physicsbench_to_lerobot.md](physicsbench_to_lerobot.md)
(general conversion guide).

## TL;DR

| | |
|---|---|
| Training data | 100 rating-5 episodes (~72k frames @ 25 Hz) from PhysicsBench v2.1 `pb-pr-edge-slide-v1` |
| Policy | Diffusion Policy, single-task, full-res 480×640 × 4 cameras |
| Steps | 30,000 (≈ 13 epochs, batch 32) |
| Final loss | 0.0184 |
| Wall-time | ~8.6 h on one L40S 46 GB |
| **Tier-1 (action MSE on val30)** | overall MSE 0.0215; NRMSE ≈ 1.0 on translation deltas (mean-predictor level) |
| **Tier-2 (PhysicsBench rollout, 50 ep)** | **success rate 34% (17/50), mean_score 0.34 ± 0.47** |
| Cost | ~30 min raw download + ~1.5 h conversion + ~8.6 h train + ~1 h eval |

The action-MSE result and the rollout result disagree — see [§7. Caveats](#7-caveats).

---

## 1. Hardware / OS assumed

- 1× L40S (46 GB) or comparable. Smoke fits in 8 GB.
- Linux x86_64 with CUDA. PhysicsBench / robosuite / mujoco only support 64-bit Linux/macOS.
- ≥ 60 GB free disk for the converted pilot dataset + checkpoints + eval. We
  used `/data` as a separate disk mount (root was tight, see [§2.1](#21-disk-layout)).

## 2. Setup

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
    ├── train/dp_edge_slide_pilot100/  # checkpoints + wandb
    └── eval/dp_pilot100_step30000/    # action_mse + rollout results + videos
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

## 3. Data prep

### 3.1 Download raw PhysicsBench v2.1 from S3

Bucket / region in [scripts/config.conf](config.conf): `physicsbench-data`,
`ap-south-1`.

```bash
./scripts/fetch_from_s3.sh lerobot_v21/pb-pr-edge-slide-v1 ~/dataset
```

Result: `~/dataset/` ≈ 13 GB, 823 episodes (`meta/episode_*_metadata.json`),
4 cameras (ego, exo_left, exo_right, gripper) at 480×640.

### 3.2 Convert pilot-100 split (training)

```bash
/home/ubuntu/lerobot/.venv/bin/python scripts/convert_physicsbench_to_lerobot.py \
  --src-path ~/dataset \
  --dst-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot-pilot100 \
  --num-episodes 100
```

Default `--min-rating 5` → keeps the cleanest 100 of 823 episodes.

| | |
|---|---|
| Output frames | 72,367 (after 100→25 Hz subsample) |
| Wall time | ~1 h 35 min (single-process, 4 mp4 re-encodes per episode) |
| Avg | ~720 frames/ep, ~57 s/ep |

Lands at `$HF_LEROBOT_HOME/manav-robotics/pb-pr-edge-slide-v1-lerobot-pilot100/`.

### 3.3 Convert val30 split (held-out)

Carved cleanly from the next 30 rating-5 episodes after the train slice via
the `--skip-episodes` flag:

```bash
/home/ubuntu/lerobot/.venv/bin/python scripts/convert_physicsbench_to_lerobot.py \
  --src-path ~/dataset \
  --dst-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot-val30 \
  --skip-episodes 100 --num-episodes 30
```

Result: 30 episodes, 22,420 frames.

## 4. Training

```bash
wandb login   # one-time, paste API key from https://wandb.ai/authorize

/home/ubuntu/lerobot/.venv/bin/lerobot-train \
  --dataset.repo_id=manav-robotics/pb-pr-edge-slide-v1-lerobot-pilot100 \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.crop_shape='[480,640]' \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=30000 \
  --eval_freq=0 \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=true \
  --wandb.project=physicsbench \
  --wandb.notes='DP pilot, edge-slide v2.1, 100 rating-5 episodes, full-res' \
  --output_dir=/data/lerobot/outputs/train/dp_edge_slide_pilot100 \
  --job_name=dp_edge_slide_pilot100
```

### 4.1 Important: do NOT pass `--policy.scheduler_decay_steps`

`DiffusionConfig` doesn't have that field — it raises `DecodingError` at
parse time. DP uses HF diffusers' cosine scheduler with
`scheduler_warmup_steps=500` and decay automatically spans `--steps`.

### 4.2 Resume after a crash

If the trainer dies between checkpoint writes, the latest checkpoint may be
partial (missing `optimizer_param_groups.json` and/or `scheduler_state.json`
in `training_state/`). The `last` symlink correctly points at the most recent
**complete** checkpoint. To resume:

```bash
/home/ubuntu/lerobot/.venv/bin/lerobot-train \
  --config_path=/data/lerobot/outputs/train/dp_edge_slide_pilot100/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --output_dir=/data/lerobot/outputs/train/dp_edge_slide_pilot100
```

`--config_path` must point at a real local file. If you get
`HFValidationError: Repo id must be in the form…`, the path is wrong (and
LeRobot fell through to `hf_hub_download(repo_id=...)`). Check that the
file at the given path exists.

### 4.3 Result

| | |
|---|---|
| Final step | 30,000 |
| Final train loss | 0.0184 |
| Total wall time | ~8.6 h (across original run + one resume) |
| Checkpoints saved | 15000, 20000, 25000, 30000 (`last` → 30000) |
| Wandb run id | `149mp100` (project `physicsbench`) |
| Disk | ~12 GB (4 × 2.9 GB checkpoints + wandb logs) |

## 5. Tier-1 eval — action MSE on val30

Cheap screen, no sim required.

```bash
mkdir -p /data/lerobot/outputs/eval/dp_pilot100_step30000

/home/ubuntu/lerobot/.venv/bin/python scripts/eval_dp_action_mse.py \
  --policy-path /data/lerobot/outputs/train/dp_edge_slide_pilot100/checkpoints/last/pretrained_model \
  --val-repo-id manav-robotics/pb-pr-edge-slide-v1-lerobot-val30 \
  --output /data/lerobot/outputs/eval/dp_pilot100_step30000/action_mse_full.json
```

22,420 frames evaluated single-step (no chunking). NRMSE = RMSE / per-dim
ground-truth std; 1.0 ≈ "predict-the-mean" baseline.

| Dim | Meaning | RMSE | GT std | **NRMSE** |
|---|---|---|---|---|
| 0 | dx | 0.141 | 0.116 | **1.22** |
| 1 | dy | 0.144 | 0.154 | **0.93** |
| 2 | dz | 0.104 | 0.103 | **1.01** |
| 3 | droll | 0.048 | 0.060 | **0.80** |
| 4 | dpitch | 0.0013 | 0.0011 | (std≈0, ignore) |
| 5 | dyaw | 0.0008 | ~0 | (std≈0, ignore) |
| 6 | gripper | 0.311 | 0.594 | **0.52** |
| **overall** | | | | MSE 0.0215 |

Read: predictions look mean-predictor-level on translation deltas.
**This is misleading** — see [§7](#7-caveats).

## 6. Tier-2 eval — PhysicsBench rollout

### 6.1 Full 50-episode run

```bash
export PYTHONUNBUFFERED=1   # otherwise stdout buffers when piped

/home/ubuntu/lerobot/.venv/bin/python -u scripts/eval_dp_in_physicsbench.py \
  --policy-path /data/lerobot/outputs/train/dp_edge_slide_pilot100/checkpoints/last/pretrained_model \
  --task-id pb-pr-edge-slide-v1 \
  --n-episodes 50 --seed-offset 1000 \
  --max-episode-steps 1000 \
  --output /data/lerobot/outputs/eval/dp_pilot100_step30000/rollout_50ep.json
```

`--control-freq` defaults to `25` to match the dataset's 25 Hz subsample
(PhysicsBench's edge-slide default is 100 Hz; running the env at 100 Hz with a
policy trained on 25 Hz subsampled deltas overshoots — see
[physicsbench_to_lerobot.md §C](physicsbench_to_lerobot.md#tier-2--physicsbench-rollout-the-real-eval-30-45-min-for-50-eps)).

| | |
|---|---|
| Episodes | 50, seeds 1000–1049 |
| **Success rate** | **34.00% (17 / 50)** |
| Mean score | 0.340 ± 0.474 |
| Mean length | 852 / 1000 step cap |
| Wall time | 61 min (73 s/ep) |
| 95% CI on success rate | ≈ 22–48% |

Failure mode: most failures hit the 1000-step cap (`truncated=True`); a few
short-length terminations were object-fell-off-edge fails (`terminated=True`,
score=0).

### 6.2 Video capture for 5 representative seeds

```bash
/home/ubuntu/lerobot/.venv/bin/python -u scripts/eval_dp_in_physicsbench.py \
  --policy-path /data/lerobot/outputs/train/dp_edge_slide_pilot100/checkpoints/last/pretrained_model \
  --task-id pb-pr-edge-slide-v1 \
  --seeds 1044,1034,1001,1028,1049 \
  --max-episode-steps 1000 \
  --videos-dir /data/lerobot/outputs/eval/dp_pilot100_step30000/videos \
  --video-camera rgb_ego_cam \
  --output /data/lerobot/outputs/eval/dp_pilot100_step30000/rollout_videos.json
```

Output: 5 mp4s named `ep<idx>_seed<seed>_<success|fail>_len<N>.mp4` under
`videos/`. Each ~0.4–1 MB at 25 fps libx264.

Note the per-seed outcomes will not match the 50-ep run on the same seeds —
DP samples fresh denoising noise at every inference call without an
external RNG hook ([modeling_diffusion.py:243-253](../src/lerobot/policies/diffusion/modeling_diffusion.py#L243-L253)),
so each rerun is a different draw.

## 7. Caveats

1. **Action-MSE is a noisy proxy for chunked policies.** Single-step MSE
   missed a ~34%-success policy and reported "mean-predictor." DP's strength
   is the smoothed action chunk it produces over the receding horizon, which
   single-step MSE doesn't measure. Treat MSE as a directional signal only,
   not a pass/fail bar. A chunk-MSE metric would be a fairer screen.
2. **DP is non-deterministic across reruns on the same env seed.** Same
   `env.reset(seed=k)` + same checkpoint → different success outcome on
   reruns, because policy noise is sampled fresh each `select_action()`
   call. Adds ~5–8% extra noise to a 50-ep success-rate measurement on top
   of the binomial sampling noise.
3. **FPS recipe matters.** Training data is 25 Hz subsampled, env defaults
   to 100 Hz. We pass `config_overrides={"control_freq": 25}`. Running the
   env at 100 Hz without the corresponding action-recipe change will
   overshoot.
4. **No sim env registered with `lerobot-eval`.** That's why we wrote a
   separate adapter ([scripts/eval_dp_in_physicsbench.py](eval_dp_in_physicsbench.py))
   instead of using `lerobot-eval --env.type=...`.
5. **PhysicsBench `run_benchmark` works** but doesn't accept `config_overrides`,
   which is why the adapter doesn't delegate to it. Same metrics
   (`mean_score`, `success_rate (>0.9)`, `mean_length`), same scoring
   contract.
6. **Resume requires the original output_dir layout.** If you `mv` the run
   directory, keep the `dp_edge_slide_pilot100/checkpoints/<step>/...` path
   intact relative to the new root. The relative `last → 015000` symlink
   survives `mv` as long as the parent dir moves as a whole.

## 8. Artifact paths

```
/home/ubuntu/dataset/                                        ← raw v2.1 source (read-only)
/data/lerobot/cache/huggingface/lerobot/manav-robotics/
├── pb-pr-edge-slide-v1-lerobot-pilot100/                    ← train slice (100 ep)
└── pb-pr-edge-slide-v1-lerobot-val30/                       ← val slice (30 ep)
/data/lerobot/outputs/train/dp_edge_slide_pilot100/
├── checkpoints/{15000,20000,25000,030000,last}/
└── wandb/                                                   ← 3 segments under run 149mp100
/data/lerobot/outputs/eval/dp_pilot100_step30000/
├── action_mse_full.json                                     ← Tier-1 result
├── action_mse_smoke.json                                    ← (discard, 100-frame smoke)
├── rollout_smoke.json                                       ← 2-ep pipeline smoke
├── rollout_50ep.json                                        ← Tier-2 full result
├── rollout_videos.json                                      ← 5-seed video run metrics
└── videos/                                                  ← 5 ego-cam mp4s
```

## 9. Suggested next steps

| Priority | Action | Cost |
|---|---|---|
| 1 | Convert all 751 rating-5 episodes (drop `--num-episodes`) and retrain at `--steps=150000`. Most likely to move success rate up. | ~12 h end-to-end |
| 2 | Sweep checkpoints 15k/20k/25k via Tier-2 to confirm 30k is best (or detect overfit). | ~3 h |
| 3 | Add chunk-MSE to the Tier-1 script — fairer screen for DP. | small |
| 4 | Save `rgb_gripper_cam` videos in addition to `rgb_ego_cam` to diagnose grasp-phase failures. | ~7 min on 5 seeds |
| 5 | Try `--include-ft` on the converter (adds force-torque to `observation.state`, 9-dim → 15-dim) and retrain. Edge-slide is contact-rich; F/T is privileged signal already in the dataset. | ~12 h end-to-end |

## 10. Files added / changed in this run

- [scripts/convert_physicsbench_to_lerobot.py](convert_physicsbench_to_lerobot.py) — added `--skip-episodes` for clean train/val carving.
- [scripts/eval_dp_action_mse.py](eval_dp_action_mse.py) — Tier-1 action MSE (new).
- [scripts/eval_dp_in_physicsbench.py](eval_dp_in_physicsbench.py) — Tier-2 PhysicsBench rollout adapter (new). Added `--seeds`, `--videos-dir`, `--video-camera` for video capture on specific seeds.
- [scripts/physicsbench_to_lerobot.md](physicsbench_to_lerobot.md) — fleshed out §C (eval plan), removed the wrong `scheduler_decay_steps` flag from the training recipes.
- This file.
