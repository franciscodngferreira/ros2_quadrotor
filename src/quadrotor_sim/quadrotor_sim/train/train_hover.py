#!/usr/bin/env python3
"""Train PPO hover policy. Edit the variables below, then run this file."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------------------------------------------------------------------------
# Edit these
# ---------------------------------------------------------------------------

# Literature calibration: PPO waypoint-reaching benchmarks (gym-pybullet-
# drones ecosystem) converge in the millions of steps single-env; curriculum
# + success termination cut that substantially. With CONTROL_DT pacing and
# empty_train.sdf's 2ms/uncapped-RTF physics (~35-60 env steps/s measured),
# 1M steps is an overnight run, not days. Reassess at the curve.
TOTAL_TIMESTEPS = 1_000_000
MAX_STEPS_PER_EPISODE = 500
FINAL_MODEL_PATH = "quadrotor_goal_ppo_hold"  # writes quadrotor_goal_ppo_hold.zip;
# quadrotor_goal_ppo(.zip) and the old fixed-point quadrotor_hover_ppo.zip are
# preserved as the "before" models.

# Warm-start from an existing policy instead of random init (None = from
# scratch). The TERMINATE_ON_SUCCESS=False run below is not learning to fly
# from zero — checkpoints_goal/best_eval already reaches targets reliably and
# only needs to learn to STAY on them, so re-learning navigation from scratch
# would waste most of the run. Its <path>_vecnormalize.pkl is loaded alongside,
# since a fresh VecNormalize would feed the policy differently scaled
# observations than the ones it learned on.
RESUME_FROM = "checkpoints_goal/best_eval"

# Domain randomization / observation-noise toggle, passed straight to
# QuadrotorHoverEnv — flip RANDOMIZE=False to reproduce the original
# fixed-spawn, fixed-target task exactly (useful as an A/B baseline).
RANDOMIZE = True
OBS_NOISE = False  # keep off for the first randomized-goal run; ablate separately later

# Control period in SIM seconds — THE fix for the long-standing plateau.
# The original step() advanced only ~10-15ms of sim time per step (one
# executor callback: IMU@100Hz / odom@50Hz), not the 50ms the task design
# assumed. Episodes were really ~5-7s instead of 25s, making many randomized
# targets physically unreachable at 0.35 m/s, and re-sampling PPO's Gaussian
# exploration every ~10ms averaged out to near-zero net displacement through
# the velocity PID — the agent could neither reach targets nor explore its
# way to the precision/success bonuses. 0.04 = exactly two 50Hz odometry
# periods of sim time per step (25Hz control; a value that isn't a multiple
# of the 0.02s odom period would silently round up to the next sample), at
# any real_time_factor. Episodes: 500 steps x 0.04s = 20s, max travel at
# 0.35 m/s = 7m >> the worst-case 2.9m spawn-target distance. None = legacy
# spin-once timing (what quadrotor_hover_ppo.zip was trained against).
CONTROL_DT = 0.04

# End episodes as "won" after holding inside SUCCESS_THRESHOLD for
# SUCCESS_HOLD_STEPS consecutive steps (0.8s at 25Hz). Literature-backed
# (termination conditions were key to sample-efficient waypoint reaching):
# denser data once the policy gets good.
#
# Turned OFF after the first goal-reaching run exposed what it actually
# optimizes. It ends the episode 0.8s after arrival, so the policy is never
# asked to STAY: rollout/success_rate hit 0.99 while a 5-episode eval_hover.py
# run scored only 60% (mean final_dist 0.152m). Those measure different things
# — 0.99 = "reached and held for 0.8s, then the episode was cut", 60% = "still
# inside 15cm at the 20s mark". The drone reached targets fine and then drifted
# in and out of the ball for the remaining ~19s, because nothing rewarded
# staying there.
#
# With this False, episodes always run the full MAX_STEPS_PER_EPISODE and the
# per-step success_bonus/precision_bonus pay out for EVERY step spent inside
# the radius, so station-keeping is the reward-maximizing behavior. It also
# makes rollout/success_rate directly comparable to eval_hover.py, since
# is_success then reduces to "dist < threshold at episode end" — the same
# question the eval asks.
TERMINATE_ON_SUCCESS = False
SUCCESS_HOLD_STEPS = 20  # inert while TERMINATE_ON_SUCCESS is False

# Adaptive curriculum: targets start close (0.75m xy range) and expand toward
# the full task spec whenever >60% of the last 20 episodes succeed. Watch
# train.log for "[curriculum] level up" lines — the leading indicator that
# learning is actually progressing.
# Off for the warm-started station-keeping run: RESUME_FROM's policy already
# cleared this curriculum to max level (3/3, the full target spec) in the
# previous run, so restarting it at level 0 would re-teach solved navigation —
# and worse, level-ups need >60% of the last 20 episodes, which the stricter
# end-of-episode is_success (see TERMINATE_ON_SUCCESS) may not clear early on,
# stalling the run at easy targets all night. Train directly at the full spec.
CURRICULUM = False

# Potential-based progress shaping (reward += PROGRESS_COEF * (prev_dist -
# dist) each step, on top of the existing distance penalty). Added after
# diagnosing that the policy was producing real, non-trivial actions but
# making ~zero net progress toward target — the static distance-penalty
# reward doesn't directly credit the action that just closed distance,
# leaving that purely to TD bootstrapping. 0.0 = off, byte-identical to the
# reward this project has used so far. 50.0 chosen so a step's max realistic
# displacement (~0.0175m at the 0.35 m/s action scale) contributes shaping
# reward (~0.9) comparable in magnitude to the existing per-step terms —
# a starting point, tune after the first run with this enabled.
PROGRESS_COEF = 50.0

# Precision bonus (Gaussian, peaks at the target, negligible a few sigma
# away) + a discrete success bonus inside the eval success radius. Added
# after a 100-episode eval of the progress-shaped model showed 0% success
# even at a generous 0.5m tolerance, plateauing around 0.75-1.5m instead —
# the linear distance penalty gives constant marginal reward per meter
# closed everywhere, so there's no extra incentive to refine the last
# stretch once "roughly close." These two terms specifically reward the
# final approach and hitting the actual measured criterion. 0.0 = off.
PRECISION_BONUS = 5.0
PRECISION_SIGMA = 0.3
SUCCESS_BONUS = 50.0
SUCCESS_THRESHOLD = 0.15  # matches eval_hover.py's default --success-threshold

# Gates the vz/attitude penalties by proximity to target (same Gaussian shape
# as PRECISION_SIGMA above). A quadrotor physically cannot move horizontally
# without tilting, so a flat, always-on tilt/vz penalty — appropriate for the
# old fixed-point task, which only ever needed tiny corrections — now fights
# against traveling efficiently toward a target up to 2.5m away. Confirmed
# against literature: a published quadrotor-racing reward-shaping method
# (FARS) does exactly this — fast/aggressive far from the goal, precise near
# it — and separately confirms flat attitude penalties can cause a policy to
# "sacrifice roll/pitch control" it actually needs. 0.0 = off (full penalty
# always, byte-identical to prior behavior).
STABILIZATION_GATE_SIGMA = 0.5

# Crash penalty. Default (in the env) is -1000.0, matching the original
# fixed-point task, which never used VecNormalize so this was never an
# issue there. For the randomized-goal task WITH VecNormalize, a saved
# checkpoint's ret_rms.std came out ~46 — meaning every reward gets divided
# by ~46 before reaching PPO, driven by rare -1000 outliers that are
# 100-1000x larger than any non-crash reward (survival ~1, distance up to
# ~2.5, success bonus up to 50). That crushes the progress/precision shaping
# terms to a couple percent of their intended relative weight. -50 keeps
# crashing clearly the single worst outcome (symmetric with the +50 success
# bonus, the largest positive term) without distorting the normalization
# scale by orders of magnitude. Crashes are already rare in practice (the
# 100-episode eval of the precision-bonus model had zero crashes), so this
# is about fixing the normalization math, not weakening crash avoidance.
CRASH_PENALTY = -50.0

# Normalize observations + rewards with running statistics (SB3 VecNormalize).
# Added after diagnosing training instability on the randomized-goal task:
# per-episode return variance is huge (a 0.75m target vs a 2.5m target are
# very different problems), which made the critic's explained_variance swing
# wildly (-0.9 to +0.99) and value_loss spike into the hundreds/thousands —
# a textbook case VecNormalize is built to stabilize. Not used in legacy mode
# (irrelevant there — the old fixed-point task never showed this instability).
USE_VEC_NORMALIZE = RANDOMIZE

# False = simple run: only FINAL_MODEL_PATH at the end (no checkpoints/ folder)
USE_CHECKPOINTS = True

# Only used if USE_CHECKPOINTS is True
# Separate from the old "checkpoints" dir (fixed-point task) — build_checkpoint_callback's
# name_prefix is hardcoded to "quadrotor_hover" and best_eval/best_train_mean share this
# dir's root, so reusing "checkpoints" here would silently overwrite the old task's
# best_eval.zip (467.5 reward) and quadrotor_hover_25000..150000_steps.zip artifacts.
# Separate dir per run: the station-keeping run must not overwrite
# checkpoints_goal/best_eval.zip, which is both the "before" model and the
# policy RESUME_FROM warm-starts from.
CHECKPOINT_DIR = "checkpoints_goal_hold"
CHECKPOINT_FREQ = 25_000
EVAL_FREQ_ROLLOUTS = 10
N_EVAL_EPISODES = 5  # bumped from 2 — randomized targets make small eval sets noisier
EVAL_MIN_DELTA = 1.0
EVAL_MIN_TIMESTEPS = 20_480
EARLY_STOP_PATIENCE = 0  # 0 = train until TOTAL_TIMESTEPS

LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
SEED = 42  # e.g. 42 for reproducibility

TENSORBOARD_LOG = "./hover_tensorboard/"
PROGRESS_BAR = False  # needs: pip install tqdm rich

# ---------------------------------------------------------------------------

def _prefer_source_tree():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


_prefer_source_tree()

from quadrotor_sim.envs.quadrotor_hover_env import QuadrotorHoverEnv  # noqa: E402
from quadrotor_sim.train.callbacks import (  # noqa: E402
    DeterministicEvalCallback,
    EarlyStopNoImprovementCallback,
    PausePhysicsDuringUpdatesCallback,
    SaveBestRolloutMeanCallback,
    build_checkpoint_callback,
    write_training_summary,
)


def _ensure_runtime_dirs():
    ws = os.path.abspath(os.getcwd())
    os.makedirs(os.environ.setdefault("ROS_HOME", os.path.join(ws, ".ros")), exist_ok=True)
    os.makedirs(os.environ.setdefault("ROS_LOG_DIR", os.path.join(ws, "log", "ros")), exist_ok=True)
    os.makedirs(os.environ.setdefault("GZ_HOME", os.path.join(ws, ".gz")), exist_ok=True)


def _progress_bar_ok() -> bool:
    if not PROGRESS_BAR:
        return False
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401
    except ImportError:
        print("[train] PROGRESS_BAR=True but tqdm/rich not installed; continuing without bar.")
        return False
    return True


def launch_gazebo():
    print("[train] Cleaning up any existing Gazebo instances...")
    subprocess.run(["pkill", "-f", "gz sim"], capture_output=True)
    subprocess.run(["pkill", "-f", "ros2 launch"], capture_output=True)
    subprocess.run(["pkill", "-f", "parameter_bridge"], capture_output=True)
    time.sleep(2.0)

    pkg_train = os.path.dirname(os.path.abspath(__file__))
    # empty_train.sdf: 2ms physics + uncapped real_time_factor (~2x wall-clock
    # throughput, verified behavior-identical for this task). Demos/GUI/legacy
    # eval keep empty.sdf — the OLD model's callback-paced timing regresses
    # under these settings. Same <world name="empty"> inside, so all
    # /world/empty/* topics and services are unchanged.
    world_file = os.path.normpath(os.path.join(pkg_train, "..", "..", "worlds", "empty_train.sdf"))
    sdf_file = os.path.normpath(
        os.path.join(pkg_train, "..", "..", "models", "quadrotor", "quadrotor.sdf")
    )

    print("[train] Launching Gazebo...")
    gz_proc = subprocess.Popen(
        ["gz", "sim", "-s", world_file],
        env=os.environ.copy(),
        stderr=subprocess.DEVNULL,
    )

    print("[train] Waiting for Gazebo to be ready...")
    while True:
        result = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True)
        if "/world/empty/clock" in result.stdout:
            print("[train] Gazebo ready.")
            break
        time.sleep(1.0)
    time.sleep(3.0)

    print("[train] Launching bridge...")
    bridge_proc = subprocess.Popen([
        "ros2", "run", "ros_gz_bridge", "parameter_bridge",
        "/quadrotor/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        "/quadrotor/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        "/quadrotor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        "/quadrotor/enable@std_msgs/msg/Bool]gz.msgs.Boolean",
    ])
    time.sleep(2.0)

    print("[train] Pausing physics...")
    subprocess.run([
        "gz", "service", "-s", "/world/empty/control",
        "--reqtype", "gz.msgs.WorldControl",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "5000",
        "--req", "pause: true",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[train] Spawning drone at z=1.0...")
    subprocess.run([
        "ros2", "run", "ros_gz_sim", "create",
        "-name", "quadrotor", "-file", sdf_file,
        "-x", "0", "-y", "0", "-z", "1.0",
    ])

    return gz_proc, bridge_proc


def _build_callbacks(checkpoint_dir: str):
    if not USE_CHECKPOINTS:
        # Still pause physics during gradient updates — with real_time_factor
        # uncapped, the drone would otherwise drift for tens of sim-seconds on
        # the last held command every update phase.
        return [PausePhysicsDuringUpdatesCallback()], None

    callbacks = []
    best_eval_path = os.path.join(checkpoint_dir, "best_eval")
    eval_cb = DeterministicEvalCallback(
        save_path=best_eval_path,
        eval_freq_rollouts=EVAL_FREQ_ROLLOUTS,
        n_eval_episodes=N_EVAL_EPISODES,
        min_delta=EVAL_MIN_DELTA,
        min_timesteps_before_save=EVAL_MIN_TIMESTEPS,
        verbose=1,
    )
    callbacks.append(eval_cb)
    callbacks.append(
        SaveBestRolloutMeanCallback(
            save_path=os.path.join(checkpoint_dir, "best_train_mean"),
            verbose=1,
        )
    )
    if CHECKPOINT_FREQ > 0:
        callbacks.append(
            build_checkpoint_callback(
                checkpoint_dir, CHECKPOINT_FREQ, save_vecnormalize=USE_VEC_NORMALIZE
            )
        )
    if EARLY_STOP_PATIENCE > 0:
        callbacks.append(
            EarlyStopNoImprovementCallback(
                patience_rollouts=EARLY_STOP_PATIENCE,
                min_delta=EVAL_MIN_DELTA,
                eval_callback=eval_cb,
                verbose=1,
            )
        )
    # MUST stay last: pauses physics at on_rollout_end, and the eval callback
    # above drives the env at that same hook (callbacks run in list order).
    callbacks.append(PausePhysicsDuringUpdatesCallback())
    return callbacks, eval_cb


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _ensure_runtime_dirs()

    checkpoint_dir = os.path.abspath(CHECKPOINT_DIR)
    if USE_CHECKPOINTS:
        os.makedirs(checkpoint_dir, exist_ok=True)

    gz_proc, bridge_proc = launch_gazebo()
    eval_cb = None

    try:
        print("[train] Creating environment...")
        raw_env = QuadrotorHoverEnv(
            max_steps=MAX_STEPS_PER_EPISODE, randomize=RANDOMIZE, obs_noise=OBS_NOISE,
            progress_coef=PROGRESS_COEF,
            precision_bonus=PRECISION_BONUS, precision_sigma=PRECISION_SIGMA,
            success_bonus=SUCCESS_BONUS, success_threshold=SUCCESS_THRESHOLD,
            stabilization_gate_sigma=STABILIZATION_GATE_SIGMA,
            crash_penalty=CRASH_PENALTY,
            control_dt=CONTROL_DT,
            terminate_on_success=TERMINATE_ON_SUCCESS,
            success_hold_steps=SUCCESS_HOLD_STEPS,
            curriculum=CURRICULUM,
        )
        # Monitor must wrap the raw env BEFORE DummyVecEnv so SB3's episode-stats
        # tracking (rollout/ep_rew_mean in TensorBoard) reflects RAW physical
        # reward — VecNormalize normalizes the reward signal used for training,
        # but does not touch what Monitor already recorded in info["episode"].
        # (Passing a raw env straight to PPO() would have SB3 auto-wrap it in
        # Monitor, but that auto-wrap only fires for non-VecEnv envs — once we
        # wrap in DummyVecEnv ourselves for VecNormalize, we must add Monitor
        # explicitly or rollout/ep_rew_mean logging silently stops working.)
        env = DummyVecEnv([lambda: Monitor(raw_env)])
        vecnorm_src = f"{RESUME_FROM}_vecnormalize.pkl" if RESUME_FROM else None
        if USE_VEC_NORMALIZE:
            if vecnorm_src and os.path.isfile(vecnorm_src):
                # Keep the running obs/reward stats the warm-start policy was
                # trained under; a fresh VecNormalize would rescale its inputs.
                env = VecNormalize.load(vecnorm_src, env)
                env.training = True
                env.norm_reward = True
                print(f"[train] Loaded VecNormalize stats <- {vecnorm_src}")
            else:
                env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=GAMMA)

        if RESUME_FROM:
            print(f"[train] Warm-starting PPO <- {RESUME_FROM}.zip")
            model = PPO.load(
                RESUME_FROM,
                env=env,
                verbose=1,
                learning_rate=LEARNING_RATE,
                tensorboard_log=TENSORBOARD_LOG,
            )
        else:
            print("[train] Creating PPO agent...")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=LEARNING_RATE,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                seed=SEED,
                tensorboard_log=TENSORBOARD_LOG,
            )

        callbacks, eval_cb = _build_callbacks(checkpoint_dir)

        print(f"[train] Training for {TOTAL_TIMESTEPS} timesteps ...")
        if USE_CHECKPOINTS:
            print(f"[train] Checkpoints -> {checkpoint_dir}/")
        else:
            print(f"[train] No checkpoints; will save -> {FINAL_MODEL_PATH}.zip")

        learn_kwargs = {
            "total_timesteps": TOTAL_TIMESTEPS,
            "progress_bar": _progress_bar_ok(),
        }
        if callbacks:
            learn_kwargs["callback"] = callbacks
        model.learn(**learn_kwargs)

        model.save(FINAL_MODEL_PATH)
        print(f"[train] Saved {FINAL_MODEL_PATH}.zip")

        if USE_VEC_NORMALIZE:
            vecnorm_path = FINAL_MODEL_PATH + "_vecnormalize.pkl"
            env.save(vecnorm_path)
            print(f"[train] Saved VecNormalize stats -> {vecnorm_path}")
            print("[train] eval_hover.py must load this file to normalize "
                  "observations the same way, or the policy will see out-of-"
                  "distribution inputs at eval time.")

        if USE_CHECKPOINTS:
            summary = {
                "total_timesteps": int(model.num_timesteps),
                "final_model": os.path.abspath(FINAL_MODEL_PATH + ".zip"),
                "best_eval_reward": eval_cb.best_mean_reward if eval_cb else None,
                "best_eval_timesteps": eval_cb.best_timesteps if eval_cb else None,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            write_training_summary(os.path.join(checkpoint_dir, "training_summary.json"), summary)
            print("[train] For eval try: checkpoints/best_eval or", FINAL_MODEL_PATH)
        else:
            print(f"[train] Eval: python3 scripts/eval_hover.py --model {FINAL_MODEL_PATH}")

    finally:
        print("[train] Shutting down...")
        gz_proc.terminate()
        bridge_proc.terminate()
        gz_proc.wait()
        bridge_proc.wait()


if __name__ == "__main__":
    main()
