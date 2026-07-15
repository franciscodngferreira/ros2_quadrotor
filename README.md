# ros2_quadrotor

[![tests](https://github.com/franciscodngferreira/ros2_quadrotor/actions/workflows/tests.yml/badge.svg)](https://github.com/franciscodngferreira/ros2_quadrotor/actions/workflows/tests.yml)

PPO-trained randomized goal-reaching for a quadrotor in **Gazebo Sim**, wired through **ROS 2 Jazzy** and **ros_gz_bridge**. Every episode the drone flies from a randomized spawn pose to a randomly-sampled 3D target and holds station on it. The original fixed-point hover task is preserved as a legacy mode.

![Randomized goal-reaching demo](docs/demo.gif)

Three fixed legs from `scripts/record_demo.py`, 2x speed. Drone pink, target red. Trained policy, deterministic actions, no scripted trajectory. Full clip: [v0.2.0](https://github.com/franciscodngferreira/ros2_quadrotor/releases/tag/v0.2.0) · earlier fixed-point hover: [v0.1.0](https://github.com/franciscodngferreira/ros2_quadrotor/releases/tag/v0.1.0).

## Results

100 randomized episodes, `scripts/eval_hover.py --episodes 100`. Per-episode data: [`docs/eval_hold_100ep.json`](docs/eval_hold_100ep.json).

| metric | value |
|---|---|
| success (`final_dist < 0.15m` at the 20s mark) | **100/100**, 95% CI [0.963, 1.000] |
| episodes completing all 500 steps | 100/100 (zero crashes) |
| final distance | mean **0.054m**, median 0.051m, sd 0.010m |
| final distance, worst of 100 | **0.082m** — 55% of the error budget |

Spread matters more than the headline: sd 1cm across 100 spawn/target pairs, range 3.9–8.2cm. Same precision regardless of where it starts. Measured with **all reward shaping off** (`eval_hover.py` builds the env from constructor defaults), so its reward is not comparable to training's — success rate and `final_dist` are the metrics.

## Stack

Gazebo Sim → `ros_gz_bridge` → `QuadrotorHoverEnv` (Gymnasium) → PPO (`train_hover.py`).

## Task

Observation is **body-frame**: `(dx_body, dy_body, dz, vx, vy, vz, roll, pitch, sin(yaw), cos(yaw))`. Position error is rotated into the drone's heading to match `cmd_vel`'s body-frame interface, so error→action is near-identity instead of a yaw-dependent rotation the network must learn. `vx,vy,vz` arrive body-frame from the odometry plugin already.

| | |
|---|---|
| target | `±1.5m` xy, `[0.6, 2.0]m` z, ≥`0.75m` from spawn |
| spawn | `±0.3m` xy, `[0.9, 1.1]m` z, random yaw, small tilt jitter |
| action | `±0.35 m/s` linear (legacy: `±0.1`, too small to cross 2.5m in an episode) |
| arena | `z ∈ [0.05, 3.0]`, `|x|,|y| ≤ 3.0` — leaving it is a crash |
| episode | 500 steps @ `control_dt=0.04` (25Hz) = 20s |

Reward per step: `1.0 - dist - 0.2*|vz| - 0.05*(|roll|+|pitch|)`, plus optional shaping (all in `train_hover.py`):

- **Progress** (`PROGRESS_COEF=50`) — `coef * (prev_dist - dist)`. Potential-based, so it telescopes to `coef * (start - end)` and can't change what's optimal, just densifies credit assignment.
- **Precision** (`PRECISION_BONUS=5`, `SIGMA=0.3`) — Gaussian peaked at the target. The linear distance penalty pays the same per metre everywhere, so "roughly close" had no pull: a 100-episode eval of the progress-only model scored 0% even at 0.5m tolerance.
- **Success** (`SUCCESS_BONUS=50`, `THRESHOLD=0.15`) — flat bonus inside the radius, i.e. exactly what `eval_hover.py` measures.
- **Stabilization gate** (`SIGMA=0.5`) — fades the `vz`/tilt penalties far from the target, restores them near it. A quadrotor can't translate without tilting, so an always-on tilt penalty fights travel.
- **Curriculum** — targets start close and widen when >60% of the last 20 episodes succeed. Watch for `[curriculum] level up` in `train.log`.

### Findings

- **Control pacing was the bug behind a long `ep_rew_mean ≈ -200` plateau.** `step()` originally returned after one executor callback — ~10-15ms of sim time, not the ~50ms assumed — so episodes really spanned 5-7s, most targets were unreachable at 0.35 m/s, and exploration re-sampled every ~10ms averaged to zero net displacement through the velocity PID. Caught by fps forensics: SB3 reported `fps=69` against a real-time-locked sim, impossible for a true 25Hz loop. Now each `step()` spins until the odom sim-time stamp advances a full period. A hand-coded P-controller then reached 10/10 targets, proving the task solvable.
- **`crash_penalty=-1000` silently crushed the shaping.** A saved VecNormalize showed `ret_rms.std ≈ 46` — the rare huge outlier inflated the reward-normalization scale until progress/precision were worth a couple percent of their intended weight. Training uses `-50`, symmetric with the `+50` success bonus; `ret_rms.std` settles ~20.
- **Success-termination optimized the wrong thing.** It ended the episode 0.8s after arrival, so nothing rewarded *staying*: `rollout/success_rate` read 0.99 while a real eval scored 60% (mean final_dist 0.152m) — the drone arrived, then drifted for the remaining 19s. Both numbers were right; they asked different questions. With it off, episodes run full length, per-step bonuses make station-keeping reward-maximizing, and `is_success` collapses to the same question the eval asks.

Future work: mass/inertia/motor randomization (needs runtime SDF respawn), wind, IMU-level noise (`obs_noise` exists but is unablated), perception, multi-waypoint, SAC/TD3 comparison, Docker.

## Setup

Needs ROS 2 Jazzy, Gazebo Sim, `ros_gz_sim`, `ros_gz_bridge`, `colcon`, and a venv with `gymnasium`, `stable-baselines3`, `torch` that can see `rclpy` after sourcing ROS (`python3 -m venv --system-site-packages ~/rl_venv`).

```bash
sudo apt install ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge python3-colcon-common-extensions
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select quadrotor_sim --symlink-install
source install/setup.bash
```

Every terminal below wants `source /opt/ros/jazzy/setup.bash && source install/setup.bash`, and the Python ones also `source ~/rl_venv/bin/activate`.

## Evaluate

```bash
ros2 launch quadrotor_sim quadrotor_gui.launch.py    # terminal 1, wait ~5s for spawn
python3 scripts/eval_hover.py --episodes 100         # terminal 2
```

Defaults to `checkpoints_goal_hold/best_eval`. Checkpoints aren't in git — download `best_eval.zip` + `best_eval_vecnormalize.pkl` from [v0.2.0](https://github.com/franciscodngferreira/ros2_quadrotor/releases/tag/v0.2.0) into `checkpoints_goal_hold/`, keeping the pair together (the `.pkl` is auto-detected beside the model; a fresh `VecNormalize` would rescale observations the policy never saw). Legacy task: `--model quadrotor_hover_ppo --no-randomize`. Headless: `quadrotor.launch.py`.

## Train

```bash
python3 src/quadrotor_sim/quadrotor_sim/train/train_hover.py
```

Settings live at the top of the file. Two phases — current defaults are phase 2:

1. **Reach** — `TERMINATE_ON_SUCCESS=True`, `CURRICULUM=True` → `checkpoints_goal/`. Reaches reliably, drifts once there.
2. **Hold** — `RESUME_FROM="checkpoints_goal/best_eval"`, both `False` → `checkpoints_goal_hold/`. Warm-starts from phase 1, so it only learns to *stay*: success 0.5 → 1.0 and eval reward 18,695 → 23,093 in ~100k steps (~1h). ~44 reward/step against a 56 ceiling ≈ 80% of the episode inside the ball, the rest being the ~3.6s approach.

Training uses `worlds/empty_train.sdf` (2ms steps, uncapped RTF, ~35-60 env steps/s); everything else uses `empty.sdf` (1ms, RTF 1.0), because the legacy model's spin-once pacing is wall-clock-entangled and regresses under the training world (0.106m → 0.158m). `PausePhysicsDuringUpdatesCallback` freezes physics during gradient updates so the drone doesn't drift on a held command.

Watch in `train.log`: `success_rate` trending up, `[callback] New best eval mean` (the deterministic eval — the only number comparable to `eval_hover.py`), `[curriculum] level up` in phase 1. `ep_len_mean` is **not** a progress signal with `TERMINATE_ON_SUCCESS=False`, and reads slightly over 500 after the first eval: `DeterministicEvalCallback` drives the same env (a second one can't share the Gazebo instance) via `unwrap_env()`, so its resets bypass `Monitor`, which keeps counting the in-flight training episode. Cosmetic — one stitched transition per ~20,480 samples, and `success_rate` is unaffected.

TensorBoard: `tensorboard --logdir hover_tensorboard`.

## Test

```bash
pip install -r requirements-dev.txt && pytest
```

Runs `scripts/check_sampling_smoke.py`'s checks — sampling ranges, quaternion round-trip, reward/crash bounds, shaping, gate, body-frame transform, success-hold, curriculum — one test case each. **No ROS or Gazebo needed**, so it runs in CI on every push: the checks are pure math on a bare env built via `__new__`, and `tests/conftest.py` stubs the ROS modules the env imports but never calls here. Real ROS wins when present, so the stub can't mask a local breakage; pytest's header says which mode ran.

Needs a live sim, so not in CI: `check_env_smoke.py` and `train/test_reset_timing.py`.

## Recording the demo

```bash
ros2 launch quadrotor_sim quadrotor.launch.py                              # 1
export GALLIUM_DRIVER=d3d12                                                # 2 — see Troubleshooting
gz sim -g --gui-config src/quadrotor_sim/config/demo_gui.config
python3 scripts/record_demo.py                                             # 3
```

Four deterministic legs (unlike `eval_hover.py`'s random ones), with a red sphere on each target — visual only, no `<collision>`, spawned at runtime so the shared `empty.sdf` doesn't inherit a stale marker (`--no-target-marker` disables). An 8s countdown gives you time to click **VideoRecorder** in the GUI; click again and complete the save dialog to write the file. Manual because no headless capture path works here (see Troubleshooting).

## Layout

| Path | What |
|------|------|
| `src/quadrotor_sim/` | ROS package (model, launch, env, train) |
| `src/quadrotor_sim/config/demo_gui.config` | Stripped GUI layout + camera framing for recordings |
| `scripts/eval_hover.py` | Success rate over N randomized episodes against a live sim |
| `scripts/record_demo.py` | Four fixed legs + target marker, for demo videos |
| `scripts/check_env_smoke.py` | Gymnasium `check_env` + randomized-reset regressions |
| `scripts/check_sampling_smoke.py` | Zero-Gazebo checks (sampling, quaternions, reward bounds) |
| `scripts/list_checkpoints.py` | List saved checkpoints |
| `tests/` | pytest wrapper around the zero-Gazebo checks |

## Topics

| ROS topic | Type |
|-----------|------|
| `/quadrotor/cmd_vel` | `geometry_msgs/Twist` |
| `/quadrotor/enable` | `std_msgs/Bool` |
| `/quadrotor/odom` | `nav_msgs/Odometry` |
| `/quadrotor/imu` | `sensor_msgs/Imu` |

## Troubleshooting

- **No odom/imu** — bridge or spawn not up; check `ros2 topic list`.
- **`rclpy` / SB3 missing** — source ROS + `install/setup.bash`, activate the venv, in that order.
- **Drone invisible in GUI** — use `quadrotor_gui.launch.py`, not the headless launch.
- **GUI at <1 fps (WSL)** — Mesa falls back to the `llvmpipe` software rasterizer (no `/dev/dri`, no accelerated DRI3 under Xwayland). Fix: `export GALLIUM_DRIVER=d3d12` before `gz sim -g`. `MESA_LOADER_DRIVER_OVERRIDE` does *not* work despite being the usual advice. Verify with `grep GL_RENDERER ~/.gz/rendering/ogre2.log | tail -1` — want `D3D12`, not `llvmpipe`. Ignore the `DRI3 error` warning; it appears either way.
- **Headless rendering is blank (WSL)** — camera sensors render empty frames and x11grab/mss capture black (WSLg composites through Wayland). The GUI renders fine via `/dev/dxg` but exposes no gz-transport service to script it, hence manual capture. A real DRI node would allow the camera-sensor route.
- **Training vs launch conflict** — run only one of `train_hover.py` or `ros2 launch` at a time.
- **Spawn/target identical every episode** — check `randomize=True` reaches `QuadrotorHoverEnv(...)`; the default differs between task versions.

## License

See `src/quadrotor_sim/package.xml`.
