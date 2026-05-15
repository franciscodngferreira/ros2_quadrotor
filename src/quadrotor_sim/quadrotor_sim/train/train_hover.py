#!/usr/bin/env python3
"""Train PPO hover policy. Edit the variables below, then run this file."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Edit these
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = 100_000
MAX_STEPS_PER_EPISODE = 500
FINAL_MODEL_PATH = "quadrotor_hover_ppo"  # writes quadrotor_hover_ppo.zip

# False = simple run: only FINAL_MODEL_PATH at the end (no checkpoints/ folder)
USE_CHECKPOINTS = False

# Only used if USE_CHECKPOINTS is True
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 25_000
EVAL_FREQ_ROLLOUTS = 5
N_EVAL_EPISODES = 4
EVAL_MIN_DELTA = 1.0
EVAL_MIN_TIMESTEPS = 20_480
EARLY_STOP_PATIENCE = 15  # 0 = train until TOTAL_TIMESTEPS

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
    world_file = os.path.normpath(os.path.join(pkg_train, "..", "..", "worlds", "empty.sdf"))
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
        return [], None

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
        callbacks.append(build_checkpoint_callback(checkpoint_dir, CHECKPOINT_FREQ))
    if EARLY_STOP_PATIENCE > 0:
        callbacks.append(
            EarlyStopNoImprovementCallback(
                patience_rollouts=EARLY_STOP_PATIENCE,
                min_delta=EVAL_MIN_DELTA,
                eval_callback=eval_cb,
                verbose=1,
            )
        )
    return callbacks, eval_cb


def main():
    _ensure_runtime_dirs()

    checkpoint_dir = os.path.abspath(CHECKPOINT_DIR)
    if USE_CHECKPOINTS:
        os.makedirs(checkpoint_dir, exist_ok=True)

    gz_proc, bridge_proc = launch_gazebo()
    eval_cb = None

    try:
        print("[train] Creating environment...")
        env = QuadrotorHoverEnv(max_steps=MAX_STEPS_PER_EPISODE)

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
