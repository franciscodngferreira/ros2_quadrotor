#!/usr/bin/env python3
"""Run a trained PPO hover policy against a live sim (GUI or headless launch)."""

import argparse
import os
import sys
import time


def _prefer_source_tree():
    ws_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'src', 'quadrotor_sim')
    )
    if ws_src not in sys.path:
        sys.path.insert(0, ws_src)


def _ensure_runtime_dirs():
    ws = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.environ.setdefault('ROS_HOME', os.path.join(ws, '.ros'))
    os.environ.setdefault('ROS_LOG_DIR', os.path.join(ws, 'log', 'ros'))
    os.environ.setdefault('GZ_HOME', os.path.join(ws, '.gz'))


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate quadrotor_hover_ppo against a running Gazebo stack.',
    )
    parser.add_argument(
        '--model',
        default='quadrotor_hover_ppo',
        help='SB3 model path (with or without .zip extension)',
    )
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument(
        '--max-steps',
        type=int,
        default=500,
        help='Steps per episode before truncation (default: 500, ~25 s at 20 Hz)',
    )
    parser.add_argument(
        '--step-sleep',
        type=float,
        default=0.0,
        help='Extra sleep (seconds) after each step for slower playback while recording',
    )
    args = parser.parse_args()

    _ensure_runtime_dirs()
    _prefer_source_tree()

    from stable_baselines3 import PPO  # noqa: E402
    from quadrotor_sim.envs.quadrotor_hover_env import QuadrotorHoverEnv  # noqa: E402

    model_path = args.model
    if not model_path.endswith('.zip'):
        model_path = model_path + '.zip'
    if not os.path.isfile(model_path):
        print(f"[eval] Model not found: {model_path}")
        print("[eval] Train first or pass --model /path/to/policy")
        sys.exit(1)

    print(f"[eval] Loading {model_path} ...")
    print("[eval] Waiting for /quadrotor/odom (start quadrotor_gui.launch.py first) ...")

    env = QuadrotorHoverEnv(max_steps=args.max_steps)
    # Load without env= to avoid Monitor/DummyVecEnv wrapper messages on stdout.
    model = PPO.load(model_path)

    # __init__ already waited for sensors; first reset() skips teleport and can
    # leave the drone settled on the ground when attaching to an external launch.
    env._first_reset = False
    obs, _ = env.reset()

    for ep in range(args.episodes):
        if ep > 0:
            obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if args.step_sleep > 0:
                time.sleep(args.step_sleep)

        z_err = float(obs[2])
        print(
            f"[eval] episode {ep + 1}/{args.episodes}: "
            f"reward={total_reward:.1f} steps={steps} final_z_err={z_err:.3f} m"
        )

    print("[eval] Done.")
    env.close()


if __name__ == '__main__':
    main()
