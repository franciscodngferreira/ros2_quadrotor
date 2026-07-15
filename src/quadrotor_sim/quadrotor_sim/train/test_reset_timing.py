#!/usr/bin/env python3
"""
Timing diagnostic for the randomized-pose reset sequence in QuadrotorHoverEnv.

Unlike the original version of this script (which reimplemented a standalone
pause/reset/unpause loop against a single fixed point using the old
`reset: {all: true}` service), this drives the ACTUAL QuadrotorHoverEnv
end-to-end — same _reset_pose_only/_at_spawn_pose/_wait_for_obs code path
used in training — across both explicit edge-case poses (spawn-range
corners, yaw=+-pi) and random samples, to confirm the pose-relative
_at_spawn_pose generalization (added for randomized spawn/target support)
converges as reliably as the original fixed-point teleport did.

Run standalone: launch Gazebo first (e.g. `ros2 launch quadrotor_sim
quadrotor.launch.py gz_args:="-s -r"`), then run this script.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from quadrotor_sim.envs.quadrotor_hover_env import QuadrotorHoverEnv  # noqa: E402


def timed_reset(env, forced_pose=None):
    if forced_pose is not None:
        original = env._sample_spawn_pose
        env._sample_spawn_pose = lambda: forced_pose
    start = time.time()
    env.reset()
    elapsed = time.time() - start
    if forced_pose is not None:
        env._sample_spawn_pose = original
    return elapsed


def main():
    env = QuadrotorHoverEnv(randomize=True)
    env.reset()  # first reset — fast path, not timed

    edge_cases = [
        ("xy corner +, yaw=+pi",  (env.spawn_xy_range, env.spawn_xy_range, 1.0, math.pi, 0.0, 0.0)),
        ("xy corner -, yaw=-pi",  (-env.spawn_xy_range, -env.spawn_xy_range, 1.0, -math.pi, 0.0, 0.0)),
        ("z low, max tilt",       (0.0, 0.0, env.spawn_z_range[0], 0.0, env.spawn_tilt_jitter, env.spawn_tilt_jitter)),
        ("z high, max tilt",      (0.0, 0.0, env.spawn_z_range[1], 0.0, -env.spawn_tilt_jitter, -env.spawn_tilt_jitter)),
        ("yaw=+pi/2",             (0.0, 0.0, 1.0, math.pi / 2, 0.0, 0.0)),
    ]

    print(f"\n--- Edge-case randomized-pose resets ({len(edge_cases)} cases) ---")
    edge_times = []
    for label, pose in edge_cases:
        elapsed = timed_reset(env, forced_pose=pose)
        edge_times.append(elapsed)
        print(f"  {label}: reset took {elapsed:.2f}s, target={env.target_xyz}")

    n_random = 15
    print(f"\n--- {n_random} random-sample resets (default ranges) ---")
    random_times = []
    for i in range(n_random):
        elapsed = timed_reset(env)
        random_times.append(elapsed)
        print(f"  reset {i}: {elapsed:.2f}s, target={env.target_xyz}")

    all_times = edge_times + random_times
    print(
        f"\n--- Summary over {len(all_times)} randomized resets ---\n"
        f"  min={min(all_times):.2f}s  mean={sum(all_times)/len(all_times):.2f}s  "
        f"max={max(all_times):.2f}s"
    )
    print("(No timeout warnings above means _at_spawn_pose converged every time.)")

    env.close()


if __name__ == "__main__":
    main()
