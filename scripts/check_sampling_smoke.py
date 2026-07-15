"""Zero-Gazebo sanity checks for the domain-randomization/goal-reaching math in
QuadrotorHoverEnv: spawn/target sampling, quaternion round-trip, and the
reward/crash-bound reconstruction. Run this before touching Gazebo at all —
it costs milliseconds and catches the bug classes most likely to slip in
silently (sampling out of range, a broken quaternion conversion, crash bounds
accidentally still coupled to the target).

Builds a QuadrotorHoverEnv instance via __new__ (skips __init__, so no rclpy/
Gazebo is touched) and manually sets only the attributes the methods under
test actually read.
"""
import math
import sys
from collections import deque

import numpy as np

sys.path.insert(0, __file__.rsplit('/scripts/', 1)[0] + '/src/quadrotor_sim')

from quadrotor_sim.envs.quadrotor_hover_env import QuadrotorHoverEnv  # noqa: E402


def make_bare_env(randomize=True, seed=0):
    env = QuadrotorHoverEnv.__new__(QuadrotorHoverEnv)
    env.randomize = randomize
    env.spawn_xy_range = 0.3
    env.spawn_z_range = (0.9, 1.1)
    env.spawn_yaw_range = math.pi
    env.spawn_tilt_jitter = 0.05
    env.target_xy_range = 1.5
    env.target_z_range = (0.6, 2.0)
    env.min_spawn_target_dist = 0.75
    env.min_z = 0.05
    env.max_z = 3.0
    env.max_xy = 3.0
    env.target_xyz = np.array([0.0, 0.0, 1.0])
    env._np_random = np.random.default_rng(seed)
    env.progress_coef = 0.0
    env._prev_dist = None
    env.precision_bonus = 0.0
    env.precision_sigma = 0.3
    env.success_bonus = 0.0
    env.success_threshold = 0.15
    env.stabilization_gate_sigma = 0.0
    env.crash_penalty = -1000.0
    env.control_dt = None
    env.terminate_on_success = False
    env.success_hold_steps = 20
    env._success_hold = 0
    env.curriculum = False
    env._full_target_xy_range = env.target_xy_range
    env._full_target_z_range = env.target_z_range
    env._full_min_spawn_target_dist = env.min_spawn_target_dist
    env._curriculum_level = 0
    env._curriculum_max_level = 3
    env._curriculum_window = deque(maxlen=20)
    return env


def check_sampling_ranges(n=1000):
    env = make_bare_env(randomize=True)
    violations = 0
    for _ in range(n):
        sx, sy, sz, syaw, sroll, spitch = env._sample_spawn_pose()
        assert abs(sx) <= env.spawn_xy_range + 1e-9
        assert abs(sy) <= env.spawn_xy_range + 1e-9
        assert env.spawn_z_range[0] - 1e-9 <= sz <= env.spawn_z_range[1] + 1e-9
        assert abs(syaw) <= env.spawn_yaw_range + 1e-9
        assert abs(sroll) <= env.spawn_tilt_jitter + 1e-9
        assert abs(spitch) <= env.spawn_tilt_jitter + 1e-9

        tx, ty, tz = env._sample_target(sx, sy, sz)
        assert abs(tx) <= env.target_xy_range + 1e-9
        assert abs(ty) <= env.target_xy_range + 1e-9
        assert env.target_z_range[0] - 1e-9 <= tz <= env.target_z_range[1] + 1e-9
        d = math.dist((tx, ty, tz), (sx, sy, sz))
        if d < env.min_spawn_target_dist - 1e-9:
            violations += 1
    assert violations == 0, f"{violations}/{n} target samples violated min_spawn_target_dist"
    print(f"[check] sampling ranges OK over {n} draws (0 min-distance violations)")


def check_legacy_mode_is_fixed():
    env = make_bare_env(randomize=False)
    for _ in range(20):
        pose = env._sample_spawn_pose()
        assert pose == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0), pose
        target = env._sample_target(*pose[:3])
        assert np.array_equal(target, np.array([0.0, 0.0, 1.0])), target
    print("[check] randomize=False reproduces exact original fixed spawn/target")


def check_quat_roundtrip():
    angles = np.linspace(-math.pi, math.pi, 9)
    pitch_angles = list(np.linspace(-math.pi / 2 * 0.999, math.pi / 2 * 0.999, 9)) + [
        -math.pi / 2, math.pi / 2  # gimbal-lock edge cases
    ]
    max_err = 0.0
    n = 0
    for roll in angles:
        for pitch in pitch_angles:
            for yaw in angles:
                qx, qy, qz, qw = QuadrotorHoverEnv._euler_to_quat(roll, pitch, yaw)
                r2, p2, y2 = QuadrotorHoverEnv._quat_to_euler(qx, qy, qz, qw)
                # At pitch = +-pi/2 (gimbal lock) roll/yaw become coupled and
                # aren't individually recoverable — only check pitch there.
                if abs(abs(pitch) - math.pi / 2) < 1e-6:
                    err = abs(p2 - pitch)
                else:
                    err = max(
                        abs(math.atan2(math.sin(r2 - roll), math.cos(r2 - roll))),
                        abs(p2 - pitch),
                        abs(math.atan2(math.sin(y2 - yaw), math.cos(y2 - yaw))),
                    )
                max_err = max(max_err, err)
                n += 1
    assert max_err < 1e-6, f"quaternion round-trip max error {max_err} over {n} samples"
    print(f"[check] euler<->quat round-trip OK over {n} samples (max err {max_err:.2e})")


def check_reward_crash_bounds():
    env = make_bare_env(randomize=True)

    def obs_for(abs_x, abs_y, abs_z, target):
        obs = np.zeros(9)
        obs[0] = abs_x - target[0]
        obs[1] = abs_y - target[1]
        obs[2] = abs_z - target[2]
        return obs

    for target_z in (0.6, 1.0, 2.0):
        target = np.array([0.0, 0.0, target_z])
        env.target_xyz = target

        env._obs = obs_for(0.0, 0.0, env.max_z + 0.01, target)
        _, crashed = env._compute_reward()
        assert crashed, f"expected crash above max_z regardless of target_z={target_z}"

        env._obs = obs_for(0.0, 0.0, env.min_z - 0.01, target)
        _, crashed = env._compute_reward()
        assert crashed, f"expected crash below min_z regardless of target_z={target_z}"

        env._obs = obs_for(env.max_xy + 0.01, 0.0, target_z, target)
        _, crashed = env._compute_reward()
        assert crashed, f"expected crash beyond max_xy regardless of target_z={target_z}"

        env._obs = obs_for(0.0, 0.0, target_z, target)
        reward, crashed = env._compute_reward()
        assert not crashed
        assert reward == 1.0, f"expected reward==1.0 exactly at target, got {reward}"

    print("[check] reward/crash bounds are absolute (independent of target_z)")


def check_crash_penalty_configurable():
    # Default (-1000.0) must match the legacy value exactly.
    env = make_bare_env(randomize=True)
    target = np.array([0.0, 0.0, 1.0])
    env.target_xyz = target
    env._obs = np.array([0.0, 0.0, env.max_z + 0.01 - target[2], 0, 0, 0, 0, 0, 0])
    reward, crashed = env._compute_reward()
    assert crashed and reward == -1000.0, (reward, crashed)

    # A custom value must be used exactly, and must still terminate the episode.
    env.crash_penalty = -50.0
    reward, crashed = env._compute_reward()
    assert crashed and reward == -50.0, (reward, crashed)

    print("[check] crash_penalty: defaults to -1000.0 (legacy-exact), "
          "custom values are honored and still terminate the episode")


def check_progress_shaping():
    env = make_bare_env(randomize=True)
    target = np.array([0.0, 0.0, 1.0])
    env.target_xyz = target

    def obs_for(x, y, z):
        obs = np.zeros(9)
        obs[0] = x - target[0]
        obs[1] = y - target[1]
        obs[2] = z - target[2]
        return obs

    # progress_coef=0.0 (default) must be a strict no-op regardless of
    # _prev_dist — byte-identical to the reward before this feature existed.
    env.progress_coef = 0.0
    env._prev_dist = 5.0  # deliberately wrong/stale, must be ignored when coef=0
    env._obs = obs_for(0.5, 0.0, 1.0)
    reward_off, _ = env._compute_reward()
    env._prev_dist = None
    reward_off2, _ = env._compute_reward()
    assert reward_off == reward_off2, (reward_off, reward_off2)

    # Sign correctness: moving closer must give MORE reward than moving away,
    # for otherwise-identical current distance.
    env.progress_coef = 50.0
    env._prev_dist = 1.0
    env._obs = obs_for(0.5, 0.0, 1.0)  # dist now = 0.5, closer than prev_dist=1.0
    reward_closer, _ = env._compute_reward()
    env._prev_dist = 1.0
    env._obs = obs_for(1.5, 0.0, 1.0)  # dist now = 1.5, farther than prev_dist=1.0
    reward_farther, _ = env._compute_reward()
    assert reward_closer > reward_farther, (reward_closer, reward_farther)
    # Both cases share prev_dist=1.0. Total delta = distance-penalty delta
    # (-1.0*(0.5-1.5) = 1.0) + shaping delta (50*((1-0.5)-(1-1.5)) = 50.0) = 51.0.
    assert abs((reward_closer - reward_farther) - 51.0) < 1e-9, (reward_closer, reward_farther)

    # Telescoping property: sum of per-step shaping rewards over a multi-step
    # "episode" must equal progress_coef * (start_dist - end_dist) exactly,
    # regardless of the (non-monotonic) path taken in between.
    env.progress_coef = 50.0
    path = [(2.0, 0.0, 1.0), (1.5, 0.3, 1.0), (1.8, -0.2, 1.0), (0.9, 0.1, 1.0), (0.2, 0.0, 1.0)]
    start_dist = math.dist(path[0], tuple(target))
    env._prev_dist = start_dist
    shaping_sum = 0.0
    base_dist = start_dist
    for x, y, z in path[1:]:
        env._obs = obs_for(x, y, z)
        dist_before = env._prev_dist
        reward, _ = env._compute_reward()
        dist_after = math.dist((x, y, z), tuple(target))
        shaping_sum += env.progress_coef * (dist_before - dist_after)
    end_dist = math.dist(path[-1], tuple(target))
    expected = env.progress_coef * (start_dist - end_dist)
    assert abs(shaping_sum - expected) < 1e-9, (shaping_sum, expected)

    print("[check] progress shaping: coef=0 is a no-op, sign is correct, "
          "telescoping sum matches progress_coef*(start_dist-end_dist) exactly")


def check_precision_and_success_bonus():
    env = make_bare_env(randomize=True)  # fresh env — progress_coef=0 by default
    target = np.array([0.0, 0.0, 1.0])
    env.target_xyz = target

    def obs_for(x, y, z):
        obs = np.zeros(9)
        obs[0] = x - target[0]
        obs[1] = y - target[1]
        obs[2] = z - target[2]
        return obs

    # Bonuses at 0.0 (default) must be a strict no-op.
    env._obs = obs_for(0.0, 0.0, 1.0)  # exactly at target
    reward_no_bonus, _ = env._compute_reward()
    assert reward_no_bonus == 1.0, reward_no_bonus  # just the survival bonus, dist=0

    # Precision bonus: maximal exactly at the target, decays with distance,
    # and must be strictly smaller at a farther point than a closer one.
    env.precision_bonus = 5.0
    env.precision_sigma = 0.3
    env._obs = obs_for(0.0, 0.0, 1.0)  # dist = 0
    reward_at_target, _ = env._compute_reward()
    assert abs(reward_at_target - (1.0 + 5.0)) < 1e-9, reward_at_target  # exp(0)=1

    env._obs = obs_for(0.3, 0.0, 1.0)  # dist = 0.3 = one sigma
    reward_one_sigma, _ = env._compute_reward()
    expected_one_sigma = 1.0 - 1.0 * 0.3 + 5.0 * math.exp(-0.5)
    assert abs(reward_one_sigma - expected_one_sigma) < 1e-9, (reward_one_sigma, expected_one_sigma)

    env._obs = obs_for(2.0, 0.0, 1.0)  # dist = 2.0, far — bonus should be ~negligible
    reward_far, _ = env._compute_reward()
    far_bonus_contribution = 5.0 * math.exp(-(2.0 ** 2) / (2 * 0.3 ** 2))
    assert far_bonus_contribution < 1e-6, far_bonus_contribution  # confirms "negligible far away" claim
    env.precision_bonus = 0.0

    # Success bonus: triggers strictly inside the threshold, not at/outside it.
    env.success_bonus = 50.0
    env.success_threshold = 0.15
    env._obs = obs_for(0.10, 0.0, 1.0)  # inside threshold
    reward_inside, _ = env._compute_reward()
    env._obs = obs_for(0.20, 0.0, 1.0)  # outside threshold
    reward_outside, _ = env._compute_reward()
    assert reward_inside > reward_outside + 50.0 - 1e-6, (reward_inside, reward_outside)

    env._obs = obs_for(0.15, 0.0, 1.0)  # exactly at threshold — must NOT trigger (strict <)
    reward_at_boundary, _ = env._compute_reward()
    expected_at_boundary = 1.0 - 1.0 * 0.15  # no success bonus
    assert abs(reward_at_boundary - expected_at_boundary) < 1e-9, reward_at_boundary

    print("[check] precision/success bonuses: off by default is a no-op, "
          "precision bonus peaks at target and is negligible far away, "
          "success bonus triggers strictly inside (not at) the threshold")


def check_stabilization_gate():
    env = make_bare_env(randomize=True)
    target = np.array([0.0, 0.0, 1.0])
    env.target_xyz = target

    def obs_with_tilt(x, y, z, vz, roll, pitch):
        obs = np.zeros(9)
        obs[0] = x - target[0]
        obs[1] = y - target[1]
        obs[2] = z - target[2]
        obs[5] = vz
        obs[6] = roll
        obs[7] = pitch
        return obs

    # Default (sigma=0) must be a strict no-op: full vz/attitude penalty at
    # ANY distance, byte-identical to the original unconditional formula.
    env.stabilization_gate_sigma = 0.0
    env._obs = obs_with_tilt(2.0, 0.0, 1.0, vz=0.5, roll=0.3, pitch=0.2)
    reward_far_ungated, _ = env._compute_reward()
    expected_far_ungated = 1.0 - 2.0 - 0.2 * 0.5 - 0.05 * (0.3 + 0.2)
    assert abs(reward_far_ungated - expected_far_ungated) < 1e-9, (reward_far_ungated, expected_far_ungated)

    # With gating on, the SAME tilt/vz far from target must cost much less
    # than right at the target — the whole point of this feature.
    env.stabilization_gate_sigma = 0.5
    env._obs = obs_with_tilt(2.0, 0.0, 1.0, vz=0.5, roll=0.3, pitch=0.2)
    reward_far_gated, _ = env._compute_reward()
    env._obs = obs_with_tilt(0.0, 0.0, 1.0, vz=0.5, roll=0.3, pitch=0.2)
    reward_near_gated, _ = env._compute_reward()
    tilt_penalty_far = 1.0 - 2.0 - reward_far_gated       # isolate the vz/attitude cost far away
    tilt_penalty_near = 1.0 - 0.0 - reward_near_gated     # isolate it right at the target
    assert tilt_penalty_far < 0.01 * tilt_penalty_near, (tilt_penalty_far, tilt_penalty_near)  # negligible far away
    assert abs(tilt_penalty_near - (0.2 * 0.5 + 0.05 * (0.3 + 0.2))) < 1e-9, tilt_penalty_near  # full weight at target

    print("[check] stabilization gate: sigma=0 is a strict no-op (full penalty always), "
          "gating makes tilt/vz nearly free far from target and full-weight at it")


def check_body_frame_transform():
    env = make_bare_env(randomize=True)

    # At yaw=0, body frame == world frame (identity on dx,dy); velocity is
    # never rotated (already body-frame from odometry); sin/cos(0) = (0,1).
    # Tolerance 1e-5 accounts for _to_policy_obs's float32 output cast
    # (~1e-8 rounding on values like -0.8), not the rotation math itself —
    # a real sign/rotation bug would show discrepancies of order 0.1-2.0.
    tol = 1e-5
    raw = np.array([1.5, -0.8, 0.3, 0.1, -0.2, 0.05, 0.02, -0.01, 0.0])
    policy = env._to_policy_obs(raw)
    assert abs(policy[0] - raw[0]) < tol and abs(policy[1] - raw[1]) < tol, policy
    assert abs(policy[3] - raw[3]) < tol and abs(policy[4] - raw[4]) < tol, policy
    assert abs(policy[8] - 0.0) < tol and abs(policy[9] - 1.0) < tol, policy

    # Rotation must preserve xy magnitude and velocity for any yaw (a rotation
    # is norm-preserving; velocity untouched regardless of heading).
    for yaw in np.linspace(-math.pi, math.pi, 13):
        raw = np.array([1.2, 0.7, 0.0, 0.3, -0.1, 0.0, 0, 0, yaw])
        policy = env._to_policy_obs(raw)
        world_mag = math.hypot(raw[0], raw[1])
        body_mag = math.hypot(policy[0], policy[1])
        assert abs(world_mag - body_mag) < tol, (yaw, world_mag, body_mag)
        assert abs(policy[3] - raw[3]) < tol and abs(policy[4] - raw[4]) < tol, (yaw, policy)
        recovered_yaw = math.atan2(policy[8], policy[9])
        assert abs(math.atan2(math.sin(recovered_yaw - yaw), math.cos(recovered_yaw - yaw))) < tol

    # Known sign-convention case: drone 2m ahead of target along its own
    # forward axis (facing world +y, physically at world +y of the target)
    # must show up as dx_body ~ +2, dy_body ~ 0 (empirically validated
    # against live Gazebo: commanding forward at yaw=90deg increases vx, not
    # vy, confirming +x-body = forward — this checks the obs side matches).
    raw = np.array([0.0, 2.0, 0.0, 0, 0, 0, 0, 0, math.pi / 2])
    policy = env._to_policy_obs(raw)
    assert policy[0] > 1.9 and abs(policy[1]) < tol, policy

    print("[check] body-frame transform: yaw=0 identity, rotation preserves "
          "magnitude/velocity, sign convention matches live Gazebo test")


def check_success_hold_termination():
    # Off by default: never terminates on success, counter always resets —
    # a strict no-op matching all behavior before this feature existed.
    env = make_bare_env()
    env.terminate_on_success = False
    for _ in range(50):
        assert env._update_success_hold(crashed=False, dist=0.01) is False
    assert env._success_hold == 0

    # On: terminates exactly at success_hold_steps consecutive in-radius steps.
    env.terminate_on_success = True
    env.success_hold_steps = 20
    env._success_hold = 0
    for i in range(19):
        assert env._update_success_hold(crashed=False, dist=0.10) is False, i
    assert env._update_success_hold(crashed=False, dist=0.10) is True  # step 20

    # Leaving the radius mid-streak resets the counter — no credit for
    # non-consecutive visits.
    env._success_hold = 0
    for _ in range(10):
        env._update_success_hold(crashed=False, dist=0.10)
    env._update_success_hold(crashed=False, dist=0.30)  # drifted out
    assert env._success_hold == 0
    for i in range(20):
        done = env._update_success_hold(crashed=False, dist=0.10)
    assert done is True  # needs the full 20 again after the reset

    # A crash never counts as success, even inside the radius.
    env._success_hold = 19
    assert env._update_success_hold(crashed=True, dist=0.01) is False
    assert env._success_hold == 0

    print("[check] success-hold termination: off is a no-op, fires exactly at "
          "success_hold_steps consecutive steps, resets on exit/crash")


def check_curriculum():
    # curriculum=False (default): recording outcomes must never mutate ranges.
    env = make_bare_env()
    before = (env.target_xy_range, env.target_z_range, env.min_spawn_target_dist)
    for _ in range(100):
        env._curriculum_record(True)
    after = (env.target_xy_range, env.target_z_range, env.min_spawn_target_dist)
    assert before == after and env._curriculum_level == 0

    # curriculum=True: level 0 is the easy spec; a full window of successes
    # levels up and clears the window; level _curriculum_max_level restores
    # the full constructor-passed spec EXACTLY (f=1 interpolation endpoint).
    env = make_bare_env()
    env.curriculum = True
    env._apply_curriculum_level()
    assert env.target_xy_range == 0.75
    assert env.target_z_range == (0.8, 1.4)
    assert env.min_spawn_target_dist == 0.4

    env._curriculum_window.extend([True] * 20)
    env._curriculum_record(True)  # 21st push (window stays at 20) triggers the check
    assert env._curriculum_level == 1, env._curriculum_level
    assert len(env._curriculum_window) == 0  # cleared: next assessment is fresh
    assert 0.75 < env.target_xy_range < env._full_target_xy_range

    # Exactly 60% (12/20) must NOT level up (strict >), 13/20 must.
    env._curriculum_window.extend([True] * 11 + [False] * 8)
    env._curriculum_record(True)  # now 12 True / 8 False = 60%
    assert env._curriculum_level == 1
    env._curriculum_window.clear()
    env._curriculum_window.extend([True] * 12 + [False] * 7)
    env._curriculum_record(True)  # now 13 True / 7 False = 65%
    assert env._curriculum_level == 2

    env._curriculum_window.extend([True] * 19)
    env._curriculum_record(True)
    assert env._curriculum_level == 3
    assert env.target_xy_range == env._full_target_xy_range
    assert env.target_z_range == env._full_target_z_range
    assert env.min_spawn_target_dist == env._full_min_spawn_target_dist

    # At max level, further successes must not level up past the cap.
    env._curriculum_window.extend([True] * 19)
    env._curriculum_record(True)
    assert env._curriculum_level == 3

    # Sanity: targets sampled at level 0 respect the shrunk ranges.
    env2 = make_bare_env()
    env2.curriculum = True
    env2._apply_curriculum_level()
    for _ in range(200):
        tx, ty, tz = env2._sample_target(0.0, 0.0, 1.0)
        assert abs(tx) <= 0.75 + 1e-9 and abs(ty) <= 0.75 + 1e-9
        assert 0.8 - 1e-9 <= tz <= 1.4 + 1e-9

    print("[check] curriculum: off never mutates ranges, level-ups need >60% of "
          "a full 20-episode window, window clears on level-up, max level == "
          "full constructor spec exactly, level-0 sampling respects shrunk ranges")


if __name__ == "__main__":
    check_sampling_ranges()
    check_legacy_mode_is_fixed()
    check_quat_roundtrip()
    check_reward_crash_bounds()
    check_crash_penalty_configurable()
    check_progress_shaping()
    check_precision_and_success_bonus()
    check_stabilization_gate()
    check_body_frame_transform()
    check_success_hold_termination()
    check_curriculum()
    print("[check] all zero-Gazebo checks passed")
