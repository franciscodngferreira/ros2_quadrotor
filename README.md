# ros2_quadrotor

PPO-trained randomized goal-reaching for a quadrotor in **Gazebo Sim**, wired through **ROS 2 Jazzy** and **ros_gz_bridge**. The drone must fly to a different randomly-sampled 3D target from a randomized spawn pose every episode, with optional Gaussian sensor noise — a step up from the original fixed-point hover task, which is preserved as a legacy mode (see below).

![Randomized goal-reaching demo](docs/demo.gif)

Three fixed legs from `scripts/record_demo.py`, 2x speed: the drone (pink) flies from a corner of the arena to a randomly-placed target (red dot) and holds station on it. Trained policy, deterministic actions, no scripted trajectory.

Full clip (real time, 25 fps): [v0.2.0 release](https://github.com/franciscodngferreira/ros2_quadrotor/releases/tag/v0.2.0). Earlier fixed-point hover, pre-randomization: [v0.1.0 release](https://github.com/franciscodngferreira/ros2_quadrotor/releases/tag/v0.1.0).

## Stack

Gazebo Sim → `ros_gz_bridge` → `QuadrotorHoverEnv` (Gymnasium) → PPO (`train_hover.py`).

## Task

In randomized mode, `QuadrotorHoverEnv`'s observation is **body-frame**: `(dx_body, dy_body, dz, vx, vy, vz, roll, pitch, sin(yaw), cos(yaw))` — position error rotated into the drone's own heading, matching `cmd_vel`'s body-frame interface (`comLinkName=base_link`), so the mapping from observed error to corrective action is close to identity rather than requiring the network to learn a yaw-dependent rotation. `vx,vy,vz` are already body-frame as published by the odometry plugin (empirically confirmed, not rotated again). Each episode:

- **Control timestep is paced by SIM time** (`control_dt=0.04` s, 25Hz): each `step()` spins the ROS executor until the odometry sim-time stamp has advanced a full control period. The original implementation returned after ONE executor callback — with IMU at 100Hz and odom at 50Hz, that meant ~10-15ms of sim time per step instead of the ~50ms the task design assumed. Found via fps forensics: SB3 reported `fps=69` against a real-time-locked Gazebo, which a true 25Hz control loop could never exceed. Consequences were fatal to learning and explain a long plateau at `ep_rew_mean ≈ -200`: episodes really spanned ~5-7s (not 20s), so many targets were physically unreachable at 0.35 m/s, and PPO's Gaussian exploration re-sampled every ~10ms averaged out to near-zero net displacement through the velocity PID (a documented failure mode of too-high control frequency). With the fix, a hand-coded P-controller reaches 10/10 randomized targets in 58-127 steps — the ground-truth proof the task is solvable. Legacy mode (`control_dt=None`) keeps the original spin-once pacing that `quadrotor_hover_ppo.zip` was trained against.
- **Success termination** (`terminate_on_success`, now **off** — this is the training story): with it on, the episode ends "won" after holding inside the success radius for 20 consecutive steps (0.8s), which gives denser data once the policy gets good. It also quietly optimizes the wrong thing. That run reported `rollout/success_rate = 0.99` while a 5-episode `eval_hover.py` scored **60%** (mean final_dist 0.152m). Both numbers were correct and measured different questions: 0.99 = "reached the target and held 0.8s, then the episode was cut", 60% = "still inside 15cm at the 20s mark". The drone arrived fine and then drifted in and out of the ball for the remaining ~19s, because nothing rewarded staying. Turning it **off** makes every episode run the full 500 steps so the per-step `success_bonus`/`precision_bonus` pay out for *every* step spent inside the radius — station-keeping becomes the reward-maximizing behavior — and collapses `is_success` to "alive and `dist < threshold` at episode end", the same question the eval asks. The two are now directly comparable. The env reports `info["is_success"]` at episode end, so SB3 logs `rollout/success_rate` to TensorBoard automatically.
- **Adaptive curriculum** (`curriculum`, on for the reaching phase, **off** for the station-keeping phase): targets start close (`±0.75m` xy, `[0.8, 1.4]m` z, ≥0.4m from spawn) and expand by linear interpolation toward the full spec below whenever >60% of the last 20 episodes succeed (4 levels total; watch `train.log` for `[curriculum] level up` lines — the leading indicator of learning progress). It's off for the warm-started hold run because the resumed policy had already cleared the curriculum to max level, so re-running it would only narrow the target distribution the policy already handles.

- **Target** is sampled within `±1.5m` xy / `[0.6, 2.0]m` z, at least `0.75m` from the spawn point.
- **Spawn pose** is randomized within `±0.3m` xy / `[0.9, 1.1]m` z, with a full-range random yaw and small roll/pitch jitter.
- **Action scale** is `±0.35 m/s` linear (up from the original `±0.1 m/s`, which was sized for tiny corrections near a single fixed point and left no margin to reach a target up to ~2.5m away within a 500-step episode — the first training attempt stalled because of this).
- **Reward** is `1.0 - 1.0*dist_to_target - 0.2*|vz| - 0.05*(|roll|+|pitch|)` per step, with a configurable terminal penalty (`crash_penalty`, default `-1000.0`) for leaving the arena (absolute bounds: `z ∈ [0.05, 3.0]`, `|x|,|y| ≤ 3.0`, independent of the target). The randomized-goal task uses `CRASH_PENALTY = -50.0` in `train_hover.py` instead — a saved VecNormalize checkpoint showed `ret_rms.std ≈ 46`, meaning the rare `-1000` outlier (100-1000x larger than any non-crash reward) was inflating the reward-normalization scale and crushing the progress/precision shaping terms to a couple percent of their intended weight; `-50` (symmetric with the `+50` success bonus, still clearly the worst outcome) brought a fresh run's `ret_rms.std` down to ~20 within 2048 steps. Optional **progress shaping** (`progress_coef`, off by default) adds `progress_coef * (prev_dist - dist)` — potential-based shaping that credits each step for the distance it actually closed, on top of the static distance penalty. It's policy-invariant (sums telescopically to `progress_coef * (start_dist - end_dist)` over an episode, verified in `check_sampling_smoke.py`) — it doesn't change what's optimal, just gives denser per-step credit assignment. `PROGRESS_COEF` in `train_hover.py` (default `50.0`).
- **Precision + success bonuses** (off by default): the linear distance penalty above gives constant marginal reward per meter closed everywhere, so a policy that gets "roughly close" has little incentive to refine further — a 100-episode eval of the progress-shaped model confirmed exactly this (0% success even at a generous 0.5m tolerance, plateauing around 0.75-1.5m). `precision_bonus * exp(-dist²/(2·precision_sigma²))` adds a Gaussian sharply peaked at the target (negligible at long range, dominant near it — a "final approach" incentive). `success_bonus` adds a flat bonus when `dist < success_threshold`, directly optimizing for the criterion `eval_hover.py` measures. `PRECISION_BONUS`/`PRECISION_SIGMA`/`SUCCESS_BONUS`/`SUCCESS_THRESHOLD` in `train_hover.py` (defaults `5.0`/`0.3`/`50.0`/`0.15`).
- **Stabilization gate** (off by default): the `vz`/attitude penalty terms were sized for the original fixed-point task (only ever tiny corrections needed), but a quadrotor physically can't move horizontally without tilting — a flat, always-on tilt penalty fights against traveling efficiently toward a target up to 2.5m away. `stabilization_gate_sigma > 0` fades those two penalties to ~0 far from the target (free to maneuver aggressively) and back to full weight near it (encourage a smooth, stable hold once arrived) — the same Gaussian-proximity idea as the precision bonus. Matches a published quadrotor-racing technique (fuzzy-logic reward shaping that balances fast-far/precise-near behavior the same way). `STABILIZATION_GATE_SIGMA` in `train_hover.py` (default `0.5`).
- **Observation noise** (Gaussian, off by default) can be enabled to simulate imperfect sensors.
- **Legacy mode** (`randomize=False`) exactly reproduces the original task: fixed `(0,0,1.0)` spawn/target, 9-dim world-frame/raw-yaw observation, and the original `±0.1 m/s` action scale. The old `quadrotor_hover_ppo.zip` model loads and runs unchanged in this mode.

Not yet implemented (future work): per-episode mass/inertia/motor-constant randomization (needs runtime SDF respawn), wind/force disturbances, IMU-level sensor noise, camera/lidar/GPS perception, multi-waypoint trajectories, PPO vs SAC/TD3 comparison, Docker/CI/pytest infra.

## Prerequisites

- ROS 2 Jazzy, Gazebo Sim, `ros_gz_sim`, `ros_gz_bridge`, `colcon`
- Python venv with `gymnasium`, `stable-baselines3`, `torch` (venv should see `rclpy` after sourcing ROS — e.g. `python3 -m venv --system-site-packages ~/rl_venv`)

```bash
sudo apt install ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge python3-colcon-common-extensions
```

## Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select quadrotor_sim --symlink-install
source install/setup.bash
```

## Run demo (GUI + policy)

**Terminal 1** — sim with GUI (wait ~5 s for spawn):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch quadrotor_sim quadrotor_gui.launch.py
```

**Terminal 2** — run trained policy (train first, or use your own model):

```bash
source ~/rl_venv/bin/activate
source /opt/ros/jazzy/setup.bash && source install/setup.bash
cd ~/ros2_ws
python3 scripts/eval_hover.py --model quadrotor_goal_ppo --episodes 100 --success-threshold 0.15
```

Default model: `quadrotor_hover_ppo.zip` (legacy fixed-point task — pass `--no-randomize` to match it). New randomized-goal model: `--model quadrotor_goal_ppo` or `--model checkpoints_goal/best_eval`. `--episodes 100` gives a portfolio-ready success-rate number; use fewer for a quick look.

Longer recording: `python3 scripts/eval_hover.py --max-steps 2000 --episodes 1`

Do **not** run `train_hover.py` while the launch above is running (duplicate Gazebo/bridge).

### Recording the demo GIF

`scripts/record_demo.py` flies four fixed start→target legs (deterministic, unlike `eval_hover.py`'s random ones) and spawns a red sphere on each leg's target. Three terminals:

```bash
# 1 — server + bridge + spawn
ros2 launch quadrotor_sim quadrotor.launch.py

# 2 — GUI (see the WSL note in Troubleshooting before running this)
export GALLIUM_DRIVER=d3d12
gz sim -g --gui-config src/quadrotor_sim/config/demo_gui.config

# 3 — the policy
python3 scripts/record_demo.py --model checkpoints_goal_hold/best_eval
```

There's an 8s countdown (`--countdown`) to click **VideoRecorder** in the GUI; click it again to stop and complete the save dialog — that's what writes the file. Recording is manual because no headless path works on this machine (see Troubleshooting). The target sphere is visual only — it has no `<collision>`, so the drone flies through it and it can't perturb the flight; `--no-target-marker` disables it. It's spawned at runtime rather than living in `empty.sdf`, since `eval_hover.py` shares that world and would inherit a marker parked at the last demo's target.

## Train

Edit settings at the top of `src/quadrotor_sim/quadrotor_sim/train/train_hover.py` (`TOTAL_TIMESTEPS`, `RANDOMIZE`, `OBS_NOISE`, `USE_CHECKPOINTS`, …), then:

```bash
source ~/rl_venv/bin/activate
source /opt/ros/jazzy/setup.bash && source install/setup.bash
cd ~/ros2_ws
python3 src/quadrotor_sim/quadrotor_sim/train/train_hover.py
```

The task is trained in **two phases**, and the current defaults are the second one:

1. **Reach** — `TERMINATE_ON_SUCCESS = True`, `CURRICULUM = True` → `checkpoints_goal/`. Learns to fly to an arbitrary target. Ends up reaching reliably but drifting once there (see the success-termination bullet above).
2. **Hold** (current defaults) — `RESUME_FROM = "checkpoints_goal/best_eval"`, `TERMINATE_ON_SUCCESS = False`, `CURRICULUM = False` → `checkpoints_goal_hold/`, final model `quadrotor_goal_ppo_hold.zip`. Warm-starts from phase 1 rather than relearning navigation from scratch, and only has to learn to *stay*. Its `<path>_vecnormalize.pkl` is loaded alongside the weights — a fresh `VecNormalize` would feed the policy differently-scaled observations than the ones it learned on.

Phase 2 converges fast: from the phase-1 policy, `rollout/success_rate` went 0.5 → 1.0 and deterministic eval reward 18,695 → 23,093 within ~100k steps (~1 hour), with 5/5 eval episodes running the full 500 steps and zero crashes. Reward decomposes as ~44/step against a 56/step ceiling (1 survival + 5 precision + 50 success), i.e. the drone is inside the 15cm ball for ~80% of the episode; the missing ~20% is the initial flight to the target, which takes ~3.6s of the 20s episode.

Other knobs: `RANDOMIZE = True`, `CONTROL_DT = 0.04`, `USE_CHECKPOINTS = True`. Checkpoint dirs are kept separate per experiment so earlier runs aren't overwritten. Set `RANDOMIZE = False` to reproduce the original fixed-point task as an A/B baseline.

Training launches `worlds/empty_train.sdf` — 2ms physics steps + uncapped `real_time_factor` (~35-60 env steps/s wall-clock, verified behavior-identical for this task since the env paces itself by sim time). Demos, the GUI launch, and legacy eval keep `worlds/empty.sdf` (1ms, RTF 1.0): the legacy model's spin-once pacing is wall-clock-entangled and measurably regresses under the training world's settings (0.106m → 0.158m final dist in a regression eval). During PPO's gradient updates a callback pauses physics (`PausePhysicsDuringUpdatesCallback`) so the drone doesn't drift on a held command at uncapped sim speed.

Progress signals to watch in `train.log`: `rollout/success_rate` trending up, `[callback] New best eval mean ...` lines (the deterministic 5-episode eval, and the only number comparable to `eval_hover.py`), and `[curriculum] level up` lines in phase 1. With `TERMINATE_ON_SUCCESS = False`, `rollout/ep_len_mean` stays pinned at `MAX_STEPS_PER_EPISODE` and is *not* a progress signal.

Known cosmetic artifact: `rollout/ep_len_mean` reads slightly **above** `MAX_STEPS_PER_EPISODE` (e.g. 510) after the first deterministic eval. `DeterministicEvalCallback` drives the *same* env — a second one can't share the Gazebo instance — via `unwrap_env()`, so its resets bypass the `Monitor` wrapper: `Monitor` keeps counting the training episode it thinks is still in flight and logs one episode of `(steps_before_eval + 500)`. The mean decays back toward 500 as the 100-episode buffer flushes. It costs one stitched transition per eval (~1 sample in 20,480) and does not affect `success_rate`, which is evaluated at the env's own episode end.

TensorBoard: `tensorboard --logdir hover_tensorboard`.

## Headless sim only

```bash
ros2 launch quadrotor_sim quadrotor.launch.py
```

## Layout

| Path | What |
|------|------|
| `src/quadrotor_sim/` | ROS package (model, launch, env, train) |
| `src/quadrotor_sim/config/demo_gui.config` | Stripped-down GUI layout + camera framing used for demo recordings |
| `scripts/eval_hover.py` | Evaluate policy against a live sim; reports success rate over N randomized episodes |
| `scripts/record_demo.py` | Fly four fixed legs with a target marker, for demo videos |
| `scripts/list_checkpoints.py` | List saved checkpoints |
| `scripts/check_env_smoke.py` | Gymnasium `check_env` smoke test + randomized-reset regression checks |
| `scripts/check_sampling_smoke.py` | Zero-Gazebo checks for spawn/target sampling, quaternion math, reward/crash bounds |

## Topics

| ROS topic | Type |
|-----------|------|
| `/quadrotor/cmd_vel` | `geometry_msgs/Twist` |
| `/quadrotor/enable` | `std_msgs/Bool` |
| `/quadrotor/odom` | `nav_msgs/Odometry` |
| `/quadrotor/imu` | `sensor_msgs/Imu` |

## Troubleshooting

- **No odom/imu** — bridge or spawn not up; check `ros2 topic list`.
- **`rclpy` / SB3 missing** — source ROS + `install/setup.bash`; activate RL venv.
- **Drone invisible in GUI** — use `quadrotor_gui.launch.py` (split `-s` server + `-g` client), not headless launch alone.
- **GUI renders at <1 fps (WSL)** — Mesa silently falls back to the `llvmpipe` software rasterizer, because WSL exposes no `/dev/dri` and Xwayland offers no accelerated DRI3 device. Fix: `export GALLIUM_DRIVER=d3d12` in the terminal that launches `gz sim -g`. Note `MESA_LOADER_DRIVER_OVERRIDE=d3d12` does **not** work here despite being the usual advice. Confirm which renderer Ogre actually picked — it logs it every launch: `grep GL_RENDERER ~/.gz/rendering/ogre2.log | tail -1`. You want `D3D12 (...)`, not `llvmpipe`. The `DRI3 error: Could not get DRI3 device` warning appears even when D3D12 is active; ignore it.
- **Headless rendering is blank (WSL)** — camera sensors + `CameraVideoRecorder` produce uniformly empty frames (no `/dev/dri`, under both ogre2 and ogre1), and `ffmpeg x11grab` / `mss` screen capture return pure black (WSLg composites through Wayland, so the X root window holds no readable content). The GUI's VideoRecorder button renders correctly via `/dev/dxg` but advertises no gz-transport service, so it can't be scripted. Hence manual demo capture. On a machine with a real DRI node, the camera-sensor route would run unattended.
- **Training vs launch conflict** — only one of `train_hover.py` or `ros2 launch` at a time.
- **Spawn/target look identical every episode** — check `randomize=True` is actually passed to `QuadrotorHoverEnv(...)`; the default differs between the two task versions, so verify which model/config you're running.

## License

See `src/quadrotor_sim/package.xml`.
