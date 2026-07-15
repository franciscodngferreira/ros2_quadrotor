import os
import time
import subprocess

import numpy as np
from stable_baselines3.common.env_checker import check_env

from quadrotor_sim.envs.quadrotor_hover_env import QuadrotorHoverEnv


def check_randomized_resets(env, n=8):
    """Confirm target_xyz varies across resets and the internal raw (world-
    frame) state reconstructs the pose we just teleported to (catches stale/
    swapped target_xyz bugs). Also checks the policy-facing body-frame obs
    preserves xy distance-to-target magnitude vs the raw world-frame state,
    since a rotation must not change distance (catches a broken _to_policy_obs)."""
    targets = []
    for i in range(n):
        obs, _ = env.reset()
        targets.append(tuple(env.target_xyz))
        cx, cy, cz, _ = env._last_commanded_pose
        raw = env._obs  # internal, world-frame, noise-free ground truth
        abs_pos = raw[0:3] + env.target_xyz
        assert abs(abs_pos[0] - cx) < 0.5, (abs_pos, env._last_commanded_pose)
        assert abs(abs_pos[1] - cy) < 0.5, (abs_pos, env._last_commanded_pose)
        assert abs(abs_pos[2] - cz) < 0.35, (abs_pos, env._last_commanded_pose)

        world_xy_dist = (float(raw[0]) ** 2 + float(raw[1]) ** 2) ** 0.5
        policy_xy_dist = (float(obs[0]) ** 2 + float(obs[1]) ** 2) ** 0.5
        assert abs(world_xy_dist - policy_xy_dist) < 0.3, (
            f"body-frame transform changed xy distance: world={world_xy_dist:.3f} "
            f"policy={policy_xy_dist:.3f}"
        )
        print(f"[smoke] reset {i}: target={env.target_xyz} commanded={env._last_commanded_pose}")
    assert len(set(targets)) > 1, "target_xyz did not vary across resets with randomize=True"
    print("[smoke] randomized resets OK: targets vary, raw state reconstructs commanded pose, "
          "body-frame obs preserves distance")


def check_legacy_resets(env, n=5):
    """With randomize=False, target/spawn must stay exactly the original fixed point."""
    for i in range(n):
        env.reset()
        assert np.array_equal(env.target_xyz, np.array([0.0, 0.0, 1.0])), env.target_xyz
        assert env._last_commanded_pose[:3] == (0.0, 0.0, 1.0) or i == 0, env._last_commanded_pose
    print("[smoke] legacy (randomize=False) resets OK: target/spawn stay fixed at (0,0,1.0)")


def main():
    proc = subprocess.Popen(
        ["ros2", "launch", "quadrotor_sim", "quadrotor.launch.py", "gz_args:=-s -r"],
        env=os.environ.copy(),
    )
    time.sleep(10.0)

    env = QuadrotorHoverEnv(randomize=True, obs_noise=True)
    print("[smoke] running check_env (randomize=True, obs_noise=True)")
    check_env(env, warn=True)
    print("[smoke] check_env done")
    check_randomized_resets(env)
    env.close()

    env = QuadrotorHoverEnv(randomize=False, obs_noise=False)
    check_legacy_resets(env)
    env.close()

    proc.terminate()
    proc.wait(timeout=10)


if __name__ == "__main__":
    main()

