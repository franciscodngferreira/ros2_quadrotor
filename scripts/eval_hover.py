#!/usr/bin/env python3
"""Run a trained PPO hover policy against a live sim (GUI or headless launch)."""

import argparse
import os
import pickle
import sys
import time

import numpy as np


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
        help='SB3 model path (default: quadrotor_hover_ppo.zip from training)',
    )
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument(
        '--max-steps',
        type=int,
        default=500,
        help='Steps per episode before truncation (default: 500 = 20 s at the 0.04 s control period)',
    )
    parser.add_argument(
        '--step-sleep',
        type=float,
        default=0.0,
        help='Extra sleep (seconds) after each step for slower playback while recording',
    )
    parser.add_argument(
        '--success-threshold',
        type=float,
        default=0.15,
        help='Final distance-to-target (m) below which an episode counts as success (default: 0.15)',
    )
    parser.add_argument(
        '--randomize', action='store_true', default=True,
        help='Randomize spawn pose and target per episode (default: on, matches training)',
    )
    parser.add_argument(
        '--no-randomize', dest='randomize', action='store_false',
        help='Legacy mode: fixed spawn (0,0,1.0) and fixed target (0,0,1.0), for the original hover model',
    )
    parser.add_argument(
        '--vecnormalize',
        default=None,
        help='Path to a VecNormalize .pkl (default: auto-detect <model>_vecnormalize.pkl '
             'next to --model; pass "none" to force-disable even if one is found)',
    )
    parser.add_argument(
        '--control-dt',
        type=float,
        default=None,
        help='Control period in SIM seconds per step (default: 0.04 in randomized mode, '
             'matching training; legacy --no-randomize mode always uses the original '
             'spin-once pacing regardless of this flag). Pass 0 to force legacy pacing '
             'in randomized mode.',
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
        print("[eval] Train first, then use: --model checkpoints/best_eval")
        print("[eval] Or list runs: python3 scripts/list_checkpoints.py")
        sys.exit(1)

    print(f"[eval] Loading {model_path} ...")
    print("[eval] Waiting for /quadrotor/odom (start quadrotor_gui.launch.py first) ...")

    # Pacing must match what the model was TRAINED with: the new goal model
    # holds each action for 0.04s of sim time (control_dt), the legacy model
    # was trained against raw spin-once pacing (~10-15ms/step). Mismatched
    # pacing changes the effective dynamics the policy acts on.
    if not args.randomize:
        control_dt = None
    elif args.control_dt is None:
        control_dt = 0.04
    elif args.control_dt <= 0:
        control_dt = None
    else:
        control_dt = args.control_dt
    env = QuadrotorHoverEnv(
        max_steps=args.max_steps, randomize=args.randomize, control_dt=control_dt
    )
    # Load without env= to avoid Monitor/DummyVecEnv wrapper messages on stdout.
    model = PPO.load(model_path)

    vec_normalize = None
    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        auto_path = model_path[:-4] + '_vecnormalize.pkl'  # strip trailing .zip
        if os.path.isfile(auto_path):
            vecnorm_path = auto_path
    if vecnorm_path and vecnorm_path.lower() != 'none':
        with open(vecnorm_path, 'rb') as f:
            vec_normalize = pickle.load(f)
        print(f"[eval] Loaded VecNormalize stats from {vecnorm_path} — normalizing "
              f"observations to match training.")

    def policy_obs(raw_obs):
        if vec_normalize is None:
            return raw_obs
        return vec_normalize.normalize_obs(
            raw_obs.reshape(1, -1)
        ).reshape(-1).astype(np.float32)

    # __init__ already waited for sensors; first reset() skips teleport and can
    # leave the drone settled on the ground when attaching to an external launch.
    env._first_reset = False
    obs, _ = env.reset()

    rewards, final_dists, mean_dists, successes = [], [], [], []

    for ep in range(args.episodes):
        if ep > 0:
            obs, _ = env.reset()
        total_reward = 0.0
        dist_sum = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(policy_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            dist_sum += float(np.linalg.norm(obs[0:3]))
            steps += 1
            if args.step_sleep > 0:
                time.sleep(args.step_sleep)

        final_dist = float(np.linalg.norm(obs[0:3]))
        mean_dist = dist_sum / steps if steps else float('nan')
        success = final_dist < args.success_threshold
        rewards.append(total_reward)
        final_dists.append(final_dist)
        mean_dists.append(mean_dist)
        successes.append(success)

        print(
            f"[eval] episode {ep + 1}/{args.episodes}: "
            f"reward={total_reward:.1f} steps={steps} "
            f"final_dist={final_dist:.3f} m mean_dist={mean_dist:.3f} m "
            f"{'SUCCESS' if success else 'FAIL'}"
        )

    n = args.episodes
    success_rate = 100.0 * sum(successes) / n if n else 0.0
    print(f"[eval] ---- Summary over {n} episodes (randomize={args.randomize}) ----")
    print(f"[eval] success rate (final_dist < {args.success_threshold} m): {success_rate:.1f}%")
    print(f"[eval] mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"[eval] mean final_dist: {np.mean(final_dists):.3f} m")
    print(f"[eval] mean time-avg dist: {np.mean(mean_dists):.3f} m")
    print("[eval] Done.")
    env.close()


if __name__ == '__main__':
    main()
