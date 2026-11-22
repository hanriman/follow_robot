# Recurrent Spatial PPO & Curriculum Upgrade

This document summarizes the diagnosis, architectural fixes, reward/curriculum improvements, vectorized training pipeline, and experimental results for the AckBot pure-sensor PPO target-following policy.

---

## 1. Root Causes Identified

1. **Zero Temporal Memory & Sensor Blind Spots:** `AckBotFollowEnv` outputs a single `426-D` frame (`64` camera depth rays over `85°` up to `5.0 m`, `360` LiDAR rays capped at `3.0 m`, `2` proprioception features). Once the target left the `85°` camera cone beyond `3.0 m` or walked behind a blind corner, a memoryless policy had no signal indicating target velocity or exit direction.
2. **MLP Background Saturation & Lack of Angular Structure:** Plain `Linear` + `Tanh` layers over raw `[0, 1]` ranges (where empty space is `1.0` and a human leg is only 2–4 rays) saturated hidden units and lacked angular localization.
3. **Clamped Gaussian `logstd` Runaway & Shared Trunk Interference:** Sampling from an unbounded `Normal`, clamping to `[-1, 1]`, and evaluating `dist.log_prob` at the clamped `±0.99999` boundary created an artificial gradient driving `actor_logstd` upward (`std → 0.87`), drowning out steering control; sharing one hidden trunk between actor and critic allowed `O(100)` value errors to perturb steering weights.
4. **Flat Out-of-FOV Reward Cliff & Unphysical Stage 1 Snaps:** `reward_visual` was a flat `-0.5` everywhere outside `±42.5°` (zero gradient to turn back toward a lost target), `reward_dist`/`reward_slot` paid positive rewards even when facing away, and Stage 1 boundary hits snapped `human.yaw` instantaneously in 1 frame.
5. **Single-Environment Bottleneck & Curriculum Leaks:** Scalar 424-ray Python loops and `num_envs = 1` limited throughput, while `AckBotFollowEnv` ignored `ppo_follow.yaml` curriculum thresholds and mixed replayed earlier-stage episodes into frontier graduation averages.

---

## 2. Changes Implemented

### Policy Architecture (`rl_ws/policies/ppo/actor_critic.py`)
- Preserved the exact `426-D` pure raw-sensor input contract and `3.00 m` LiDAR hardware limit.
- Added proximity inversion (`1.0 - range_norm`) and registered angular coordinate buffers (`cam_angles`, `lidar_cos`, `lidar_sin`).
- Replaced dense input MLPs with 1D `CoordConv` encoders (using `padding_mode="circular"` for 360° LiDAR) fused with differentiable polar soft-argmin and sector max-proximity summary features.
- Added `GRU(128, 128)` temporal memory with a current-frame spatial residual skip and episode-boundary hidden-state masking (`terminated | truncated`).
- Decoupled `actor_mean` and `critic` trunks and replaced clamped `Normal` with an exact `TanhNormal` squashed Gaussian (`log(1 - tanh²)` Jacobian correction, `actor_logstd` clamped to `[-2.5, -0.35]`).

### Environment, Rewards & Raycasting (`rl_ws/gym_ackbot/envs/ackbot_follow_env.py`)
- Replaced the out-of-FOV `-0.5` cliff with a continuous 360° bearing reward (`0.45 * cos(angle_to_cam) - 0.15`), added potential-based angular tracking (`1.5 * (prev_angle - |angle|)`), and gated `reward_dist`/`reward_slot` by `max(0, cos(angle_to_cam))` when unobstructed.
- Added out-of-FOV turning recovery + search speed guard (`robot_v > 0.45 m/s` penalty) and close-range overshoot braking (`target_dist < 0.65 m`).
- Smoothed Stage 1 and Stage 4 target turns using bounded-rate steering (`_steer_towards`) and damped angular velocity `omega`.
- Vectorized `_cast_rays` in NumPy across all rays simultaneously and added `set_unlocked_stage()` plus frontier-only `curriculum_history` tracking.

### Vectorized Recurrent PPO & Callers
- **`rl_ws/policies/ppo/buffer.py`:** Added `num_envs` support, separate `dones` (collision `terminated` for GAE) vs `episode_dones` (`terminated | truncated` for GRU resets), hidden-state storage, vectorized GAE, and `get_recurrent_batches()` for sequence-chunked BPTT (`seq_len = 32`).
- **`rl_ws/train.py` & `rl_ws/configs/ppo_follow.yaml`:** Configured `SyncVectorEnv` (`num_envs = 8`, `n_steps = 512`, `AutoresetMode.DISABLED` with `reset_mask` and `_VectorSafeInfoWrapper`), synchronized frontier curriculum graduation (`min_episodes_per_stage = 40`, `stage_thresholds = {1: 850, 2: 700, 3: 600}`), `ent_coef = 0.001`, `vf_coef = 0.05`, `clip_range_vf = 5.0`, deterministic eval seeds with `target_in_fov_rate` logging, and `unlocked_stage`-aware `best_model.pt` selection.
- **`rl_ws/common/agents.py`, `scripts/view_gym_env.py`, `scripts/harvest_data.py`, `ros_ws/src/ackbot_policy/ackbot_policy/policy_node.py`:** Updated all inference callers to maintain and reset GRU hidden states (`hx`), and added the missing `rl_ws.gym_ackbot.envs` registration import in `scripts/view_gym_env.py`.
- **`tests/test_policies.py` & `tests/test_curriculum_env.py`:** Added unit tests for angular sensitivity, `TanhNormal` bounds, GRU memory & episode resets, multi-env recurrent buffer/GAE, vectorized raycasting accuracy, out-of-FOV reward gradient, and frontier-only graduation.

---

## 3. Verification & Training Results

### Automated Checks
- `uv run --extra dev pytest tests/ -v` → **58 passed**
- `uv run --extra dev ruff check rl_ws/ tests/ scripts/view_gym_env.py scripts/harvest_data.py` → **All checks passed**

### Training Runs (CPU, 8 Parallel Envs)
1. **`exp_20261122_210124_recurrent_ppo_v3`** (`600,000` steps from scratch, `1,841.1s`):
   - `entropy` annealed cleanly (`1.838 → 1.742`), graduating **Stage 1 → Stage 2 → Stage 3**.
   - Eval at step `598,016`: Stage 1 `+1144.8 (92% FOV)`, Stage 2 `+950.8 (70% FOV)`.
   - 10-episode unseen-seed test (`latest_model.pt`): Stage 1 `ret=959.5, FOV 79.4%, dist=1.21m` | Stage 2 `ret=930.1, FOV 70.7%, col=0%, dist=1.74m`.
2. **`exp_20261122_222257_recurrent_ppo_s3_s4`** (`400,000` steps warm-started from `v3/best_model.pt` at Stage 3, `1,279.1s`):
   - Graduated **Stage 3 → Stage 4** around step `151k`.
   - 10-episode unseen-seed test (`best_model.pt`, step `270,336`):
     - Stage 1: `ret=634.8`, `500/500`, `FOV 69.6%`, `col 0%`, `dist 2.38m`
     - Stage 2: `ret=789.1`, `500/500`, `FOV 45.7%`, `col 0%`, `dist 1.75m`
     - Stage 3: `ret=624.5`, `500/500`, `FOV 82.2%`, `col 0%`, `dist 3.87m`
     - Stage 4: `ret=269.2`, `456.3/500`, `FOV 39.2%`, `col 10%`, `dist 2.62m`

---

## 4. Running Locally on macOS (Apple Silicon)

Because `outputs/experiments/*` is gitignored, either copy the trained `.pt` checkpoints via `scp` or train natively on Apple Silicon MPS:

```bash
# 1. Pull latest code & sync environment
git pull origin main
uv sync --extra dev

# 2. Verify tests
uv run --extra dev pytest tests/ -v

# 3. View a checkpoint (after copying .pt from VM) or train on MPS
uv run python scripts/view_gym_env.py --policy model --stage 1 --checkpoint <path/to/model.pt>
uv run python rl_ws/train.py --device mps --exp-name mac_recurrent_ppo
```
