import gymnasium as gym
import logging
from collections import deque
import numpy as np
import os
import rclpy
import rclpy.executors
import rclpy.context
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
import subprocess
import sys
import time
import math

# gz.transport13 / gz.msgs10 are Debian packages installed under
# /usr/lib/python3/dist-packages, which the project's rl_venv doesn't see
# (it was created without --system-site-packages, unlike what the README
# assumes — only /opt/ros/jazzy's own Python path gets added via PYTHONPATH
# when sourcing ROS, which doesn't cover this separate system package dir).
_GZ_DIST_PACKAGES = "/usr/lib/python3/dist-packages"
if _GZ_DIST_PACKAGES not in sys.path and os.path.isdir(_GZ_DIST_PACKAGES):
    sys.path.append(_GZ_DIST_PACKAGES)

import gz.transport13 as gz_transport
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose

logger = logging.getLogger(__name__)


class QuadrotorHoverEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 500,
        randomize: bool = True,
        obs_noise: bool = False,
        spawn_xy_range: float = 0.3,
        spawn_z_range: tuple = (0.9, 1.1),
        spawn_yaw_range: float = math.pi,
        spawn_tilt_jitter: float = 0.05,
        target_xy_range: float = 1.5,
        target_z_range: tuple = (0.6, 2.0),
        min_spawn_target_dist: float = 0.75,
        noise_std_pos: float = 0.02,
        noise_std_vel: float = 0.02,
        noise_std_angle: float = 0.01,
        progress_coef: float = 0.0,
        precision_bonus: float = 0.0,
        precision_sigma: float = 0.3,
        success_bonus: float = 0.0,
        success_threshold: float = 0.15,
        stabilization_gate_sigma: float = 0.0,
        crash_penalty: float = -1000.0,
        control_dt: float | None = None,
        terminate_on_success: bool = False,
        success_hold_steps: int = 20,
        curriculum: bool = False,
    ):
        super().__init__()

        self.target_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.max_steps = max_steps
        self.current_step = 0
        self.min_z = 0.05
        self.max_z = 3.0
        self.max_xy = 3.0

        self.randomize = os.environ.get(
            "QUADROTOR_RANDOMIZE", str(randomize)
        ).lower() in {"1", "true", "yes", "on"}
        self.obs_noise = os.environ.get(
            "QUADROTOR_OBS_NOISE", str(obs_noise)
        ).lower() in {"1", "true", "yes", "on"}
        self.spawn_xy_range = spawn_xy_range
        self.spawn_z_range = spawn_z_range
        self.spawn_yaw_range = spawn_yaw_range
        self.spawn_tilt_jitter = spawn_tilt_jitter
        self.target_xy_range = target_xy_range
        self.target_z_range = target_z_range
        self.min_spawn_target_dist = min_spawn_target_dist
        self.noise_std_pos = noise_std_pos
        self.noise_std_vel = noise_std_vel
        self.noise_std_angle = noise_std_angle
        self._last_commanded_pose = (0.0, 0.0, 1.0, 0.0)  # (x, y, z, yaw)

        # Potential-based progress shaping: reward = ... + progress_coef *
        # (prev_dist - dist), on top of the existing distance penalty. This
        # sums telescopically over an episode to exactly (start_dist -
        # end_dist) regardless of path, so it doesn't change what's optimal
        # (Ng, Harada & Russell potential-based shaping) — it just gives
        # per-step credit for whether the last action actually closed
        # distance, instead of leaving the agent to infer that purely from
        # TD bootstrapping on a static "am I currently far" signal. Default
        # 0.0 (off) preserves the exact original reward in both legacy and
        # randomized-goal modes unless explicitly enabled.
        self.progress_coef = progress_coef
        self._prev_dist = None

        # Precision bonus: reward += precision_bonus * exp(-dist^2 / (2*sigma^2)),
        # a Gaussian sharply peaked at the target. Added because the existing
        # distance penalty is LINEAR in dist, so its marginal reward per meter
        # closed is constant everywhere — there's no extra incentive to close
        # the last 50cm vs. any other 50cm. A 100-episode eval of the
        # progress-shaped model showed exactly that failure mode: 0% within
        # 0.15m (and even 0.5m), plateauing around 0.75-1.5m instead — it
        # learned to get roughly near a target but never to converge on it.
        # This term is negligible at long range (doesn't disturb coarse
        # navigation) but dominates near the target, creating a "final
        # approach" incentive the linear term can't provide. Default 0.0
        # (off) preserves prior behavior unless explicitly enabled.
        self.precision_bonus = precision_bonus
        self.precision_sigma = precision_sigma

        # Discrete success bonus: reward += success_bonus when dist <
        # success_threshold — directly optimizes for the exact criterion
        # eval_hover.py measures, complementing the smooth Gaussian above
        # (a common pairing in precision-landing RL literature: a dense
        # proximity term plus a bonus specifically inside the target radius).
        self.success_bonus = success_bonus
        self.success_threshold = success_threshold

        # Gates the vz/attitude penalties by proximity to target (same
        # Gaussian shape as precision_bonus): a quadrotor CANNOT move
        # horizontally without tilting, so a flat, always-on tilt/vz penalty
        # (fine for the old fixed-point task, which only ever needed tiny
        # corrections) actively fights against traveling efficiently toward
        # a target up to 2.5m away now. 0.0 (default) = no gating, penalty
        # applies at full weight always — byte-identical to prior behavior.
        # A positive sigma fades the penalty to ~0 far from target (free to
        # maneuver aggressively) and back to full weight near it (encourage
        # a smooth, stable hold once arrived).
        self.stabilization_gate_sigma = stabilization_gate_sigma

        # Crash penalty. Default -1000.0 preserves the original fixed-point
        # task's behavior exactly (that task never used VecNormalize, so this
        # never caused a problem there). For the randomized-goal task WITH
        # VecNormalize, a -1000 outlier — 100-1000x larger than any non-crash
        # reward (survival ~1, distance up to ~2.5, success bonus up to 50)
        # — inflates VecNormalize's running reward-variance estimate
        # (observed ret_rms.std ~46), which then divides down EVERY reward,
        # crushing the progress/precision shaping terms to a couple percent
        # of their intended relative weight. A smaller magnitude, comparable
        # to the largest positive reward component, keeps crashing clearly
        # the worst outcome without distorting the normalization scale.
        self.crash_penalty = crash_penalty

        # Control period in SIM seconds. The original step() implementation
        # spun the executor until ONE callback fired — with IMU at 100Hz and
        # odom at 50Hz that meant each env step spanned only ~10-15ms of sim
        # time, not the 50ms the whole task was designed around: episodes
        # were really ~5-7s (not 25s), so many randomized targets were
        # physically unreachable at 0.35 m/s, and PPO's Gaussian exploration
        # (re-sampled every ~10ms) averaged to near-zero net displacement
        # through the velocity PID. A positive control_dt makes step() spin
        # until the odometry sim-time stamp has advanced by control_dt —
        # exactly one control period per step regardless of real_time_factor,
        # with a guaranteed-fresh position observation. None (default)
        # preserves the legacy spin-once timing that quadrotor_hover_ppo.zip
        # was trained against.
        self.control_dt = control_dt
        self._odom_stamp = 0.0

        # Terminate the episode as "won" after the drone holds inside the
        # success radius for success_hold_steps consecutive steps. Ends
        # episodes early once the policy gets good (denser data) and directly
        # optimizes the criterion eval_hover.py measures. Off by default.
        self.terminate_on_success = terminate_on_success
        self.success_hold_steps = success_hold_steps
        self._success_hold = 0

        # Adaptive curriculum on target difficulty: start with close targets
        # and expand toward the full constructor-specified ranges as the
        # rolling success rate improves (level up when >60% of the last 20
        # episodes ended inside the success radius). curriculum=False
        # (default) uses the passed ranges as-is, unchanged behavior.
        self.curriculum = curriculum
        self._full_target_xy_range = target_xy_range
        self._full_target_z_range = target_z_range
        self._full_min_spawn_target_dist = min_spawn_target_dist
        self._curriculum_level = 0
        self._curriculum_max_level = 3
        self._curriculum_window = deque(maxlen=20)
        if self.curriculum:
            self._apply_curriculum_level()

        print(
            f"[env] max_steps={self.max_steps} randomize={self.randomize} "
            f"obs_noise={self.obs_noise} progress_coef={self.progress_coef} "
            f"precision_bonus={self.precision_bonus} precision_sigma={self.precision_sigma} "
            f"success_bonus={self.success_bonus} stabilization_gate_sigma={self.stabilization_gate_sigma} "
            f"crash_penalty={self.crash_penalty} control_dt={self.control_dt} "
            f"terminate_on_success={self.terminate_on_success} curriculum={self.curriculum} "
            f"bounds: min_z={self.min_z} max_z={self.max_z} max_xy={self.max_xy}"
        )

        self._first_reset = True

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Policy-facing observation. In randomized mode (10-dim) it's BODY-
        # frame, matching cmd_vel's body-frame interface (comLinkName=base_link)
        # so the mapping from observed error to corrective action is close to
        # identity instead of requiring the network to learn a yaw-dependent
        # rotation — see _to_policy_obs(). In legacy mode (randomize=False,
        # 9-dim, world-frame, raw yaw) the shape/frame/dtype must stay BYTE-
        # IDENTICAL to the original pre-refactor observation, since the
        # existing quadrotor_hover_ppo.zip was trained against it and SB3
        # loads a fixed-size policy network — a 10-dim obs would silently
        # break predict() with a shape mismatch. Internal self._obs (9-dim)
        # always stays world-frame/raw for reward/crash/pose-check logic.
        if self.randomize:
            obs_high = np.array([
                5.0, 5.0, 5.0,          # dx_body, dy_body, dz (position error, body-frame xy)
                5.0, 5.0, 5.0,          # vx, vy, vz (already body-frame from odometry twist)
                math.pi, math.pi/2,     # roll, pitch
                1.0, 1.0,               # sin(yaw), cos(yaw) — avoids the +-pi wrap discontinuity
            ], dtype=np.float32)
        else:
            obs_high = np.array([
                5.0, 5.0, 5.0,          # dx, dy, dz — world-frame (== original hover task)
                5.0, 5.0, 5.0,          # vx, vy, vz
                math.pi, math.pi/2, math.pi  # roll, pitch, yaw
            ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        self._obs = np.zeros(9, dtype=np.float32)
        self._odom_ready = False
        self._imu_ready = False
        self._printed_first_odom = False
        self._printed_first_imu = False
        self._debug_rx = os.environ.get("QUADROTOR_ENV_DEBUG_RX", "").lower() in {"1", "true", "yes", "on"}
        self._obs_wait_timeout = float(os.environ.get("QUADROTOR_OBS_WAIT_TIMEOUT", "20.0"))
        # After a teleport the re-armed controller only holds zero velocity, so
        # the drone often settles a little off the exact commanded height and
        # _at_spawn_pose() never converges. That is benign, so cap the settle
        # wait here instead of burning the full obs-wait timeout on every reset.
        self._spawn_settle_timeout = float(os.environ.get("QUADROTOR_SPAWN_SETTLE_TIMEOUT", "2.0"))

        # Ensure ROS logging goes somewhere writable
        ws = os.path.abspath(os.getcwd())
        ros_home = os.environ.setdefault("ROS_HOME", os.path.join(ws, ".ros"))
        ros_log_dir = os.environ.setdefault("ROS_LOG_DIR", os.path.join(ws, "log", "ros"))
        gz_home = os.environ.setdefault("GZ_HOME", os.path.join(ws, ".gz"))
        os.makedirs(ros_home, exist_ok=True)
        os.makedirs(ros_log_dir, exist_ok=True)
        os.makedirs(gz_home, exist_ok=True)

        # Own ROS context so other code can't invalidate it
        self._context = rclpy.context.Context()
        rclpy.init(context=self._context)
        self.node = rclpy.create_node('quadrotor_hover_env', context=self._context)
        self._executor = rclpy.executors.SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)

        # Persistent gz-transport client for world-control/set_pose calls,
        # reused for the env's whole lifetime. Previously each pause/teleport/
        # unpause spawned a brand-new `gz service` CLI subprocess (~3 per
        # reset), which creates and tears down a fresh gz-transport
        # participant every time — over a multi-hour run (thousands of
        # resets) this accumulated discovery-layer overhead in the
        # long-running gzserver, made basic gz-transport calls take seconds
        # instead of milliseconds, and eventually caused teleports to
        # silently fail to complete within _wait_for_obs's timeout. A single
        # persistent Node avoids the repeated participant churn entirely.
        self._gz_node = gz_transport.Node()

        self._cmd_pub = self.node.create_publisher(
            Twist, '/quadrotor/cmd_vel', 10)
        self._enable_pub = self.node.create_publisher(
            Bool, '/quadrotor/enable', 10)

        # Best-effort matches typical ros_gz_bridge sensor forwarding and avoids
        # reliable-queue stalls after pause/teleport resets.
        self._odom_sub = self.node.create_subscription(
            Odometry, '/quadrotor/odom', self._odom_cb, qos_profile_sensor_data)
        self._imu_sub = self.node.create_subscription(
            Imu, '/quadrotor/imu', self._imu_cb, qos_profile_sensor_data)

        self._enable_controller()

    # ------------------------------------------------------------------
    # ROS helpers
    # ------------------------------------------------------------------

    def _spin_once(self, timeout_sec: float):
        self._executor.spin_once(timeout_sec=timeout_sec)

    def _enable_controller(self):
        print("[env] Waiting for bridge to connect...")
        enable_msg = Bool()
        enable_msg.data = True
        cmd = Twist()

        while self.node.count_publishers('/quadrotor/enable') == 0:
            self._spin_once(timeout_sec=0.1)

        print("[env] Bridge connected, flooding enable signal...")
        start = time.time()
        while time.time() - start < 2.0:
            self._enable_pub.publish(enable_msg)
            self._cmd_pub.publish(cmd)
            self._spin_once(timeout_sec=0.05)

        print("[env] Unpausing physics...")
        self._unpause_physics()
        self._wait_for_obs(require_spawn_pose=False)
        print("[env] Controller enabled and physics running.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg):
        if self._debug_rx and not self._printed_first_odom:
            print("[env] First odom received")
            self._printed_first_odom = True
        self._odom_ready = True
        # Sim-time stamp of this odometry sample — step()'s control_dt pacing
        # keys on this advancing, which makes the control period exact in SIM
        # time regardless of real_time_factor (and guarantees the position
        # part of the observation is fresh, not a stale pre-action sample).
        self._odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self._obs[0] = pos.x - self.target_xyz[0]
        self._obs[1] = pos.y - self.target_xyz[1]
        self._obs[2] = pos.z - self.target_xyz[2]
        self._obs[3] = vel.x
        self._obs[4] = vel.y
        self._obs[5] = vel.z

    def _imu_cb(self, msg):
        if self._debug_rx and not self._printed_first_imu:
            print("[env] First imu received")
            self._printed_first_imu = True
        self._imu_ready = True
        q = msg.orientation
        roll, pitch, yaw = self._quat_to_euler(q.x, q.y, q.z, q.w)
        self._obs[6] = roll
        self._obs[7] = pitch
        self._obs[8] = yaw

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._success_hold = 0

        spawn_x, spawn_y, spawn_z, spawn_yaw, spawn_roll, spawn_pitch = self._sample_spawn_pose()
        self.target_xyz = self._sample_target(spawn_x, spawn_y, spawn_z)

        if self._first_reset:
            self._first_reset = False
            # Re-arm controller — obs already populated from __init__
            enable_msg = Bool()
            enable_msg.data = True
            start = time.time()
            while time.time() - start < 0.5:
                self._enable_pub.publish(enable_msg)
                self._spin_once(timeout_sec=0.05)
            self._prev_dist = self._current_dist()
            return self._noisy_obs(), {}

        # Subsequent resets — teleport drone back, don't delete/respawn
        self._printed_first_odom = False
        self._printed_first_imu = False

        self._reset_pose_only(spawn_x, spawn_y, spawn_z, spawn_roll, spawn_pitch, spawn_yaw)

        # Re-arm controller after teleport
        enable_msg = Bool()
        enable_msg.data = True
        start = time.time()
        while time.time() - start < 1.0:
            self._enable_pub.publish(enable_msg)
            self._cmd_pub.publish(Twist())
            self._spin_once(timeout_sec=0.05)

        self._wait_for_obs(require_spawn_pose=True)
        self._prev_dist = self._current_dist()
        return self._noisy_obs(), {}

    def _current_dist(self) -> float:
        """3D distance-to-target from the internal (clean, world-frame) state."""
        return float(math.sqrt(self._obs[0] ** 2 + self._obs[1] ** 2 + self._obs[2] ** 2))

    def _update_success_hold(self, crashed: bool, dist: float) -> bool:
        """Track consecutive steps inside the success radius; returns True
        once the streak reaches success_hold_steps (episode "won"). Always
        False when terminate_on_success is off — a strict no-op."""
        if self.terminate_on_success and not crashed and dist < self.success_threshold:
            self._success_hold += 1
        else:
            self._success_hold = 0
        return self.terminate_on_success and self._success_hold >= self.success_hold_steps

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _apply_curriculum_level(self):
        """Set target-sampling ranges by linear interpolation between an easy
        starting spec and the full constructor-passed spec, according to the
        current level (0 = easiest, _curriculum_max_level = full task)."""
        f = self._curriculum_level / self._curriculum_max_level
        start_xy, start_z, start_min_dist = 0.75, (0.8, 1.4), 0.4
        self.target_xy_range = start_xy + f * (self._full_target_xy_range - start_xy)
        self.target_z_range = (
            start_z[0] + f * (self._full_target_z_range[0] - start_z[0]),
            start_z[1] + f * (self._full_target_z_range[1] - start_z[1]),
        )
        self.min_spawn_target_dist = (
            start_min_dist + f * (self._full_min_spawn_target_dist - start_min_dist)
        )

    def _curriculum_record(self, success: bool):
        """Record an episode outcome; level up once >60% of the last 20
        episodes (a full window) succeeded. The window is cleared on level-up
        so the next assessment only counts episodes at the new difficulty."""
        if not self.curriculum:
            return
        self._curriculum_window.append(bool(success))
        if (
            self._curriculum_level < self._curriculum_max_level
            and len(self._curriculum_window) == self._curriculum_window.maxlen
            and sum(self._curriculum_window) / len(self._curriculum_window) > 0.6
        ):
            self._curriculum_level += 1
            self._curriculum_window.clear()
            self._apply_curriculum_level()
            print(
                f"[curriculum] level up -> {self._curriculum_level}/{self._curriculum_max_level}: "
                f"target_xy_range={self.target_xy_range:.2f} "
                f"target_z_range=({self.target_z_range[0]:.2f}, {self.target_z_range[1]:.2f}) "
                f"min_spawn_target_dist={self.min_spawn_target_dist:.2f}"
            )

    # ------------------------------------------------------------------
    # Randomization / observation-noise helpers
    # ------------------------------------------------------------------

    def _sample_spawn_pose(self):
        """Sample (x, y, z, yaw, roll, pitch); fixed (0,0,1.0,0,0,0) if randomize=False."""
        if not self.randomize:
            return 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        x = self.np_random.uniform(-self.spawn_xy_range, self.spawn_xy_range)
        y = self.np_random.uniform(-self.spawn_xy_range, self.spawn_xy_range)
        z = self.np_random.uniform(*self.spawn_z_range)
        yaw = self.np_random.uniform(-self.spawn_yaw_range, self.spawn_yaw_range)
        roll = self.np_random.uniform(-self.spawn_tilt_jitter, self.spawn_tilt_jitter)
        pitch = self.np_random.uniform(-self.spawn_tilt_jitter, self.spawn_tilt_jitter)
        return x, y, z, yaw, roll, pitch

    def _sample_target(self, spawn_x, spawn_y, spawn_z):
        """Sample a 3D target at least min_spawn_target_dist from the spawn point."""
        if not self.randomize:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tx, ty, tz = 0.0, 0.0, 1.0
        for _ in range(20):
            tx = self.np_random.uniform(-self.target_xy_range, self.target_xy_range)
            ty = self.np_random.uniform(-self.target_xy_range, self.target_xy_range)
            tz = self.np_random.uniform(*self.target_z_range)
            if math.dist((tx, ty, tz), (spawn_x, spawn_y, spawn_z)) >= self.min_spawn_target_dist:
                break
        return np.array([tx, ty, tz], dtype=np.float64)

    def _to_policy_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Transform internal world-frame/raw state (9-dim) into the 10-dim
        body-frame, sin/cos-yaw observation the policy sees.

        Only dx, dy (position error, derived from world-frame odometry minus
        target) are rotated into body frame — vx, vy, vz are NOT rotated
        again, since they're already body-frame as published by this model's
        odometry (empirically confirmed: commanding "forward" at yaw=90deg
        shows up as vx, not vy). dz is untouched (yaw rotation doesn't mix
        into the vertical axis)."""
        dx, dy, dz = raw_obs[0], raw_obs[1], raw_obs[2]
        vx, vy, vz = raw_obs[3], raw_obs[4], raw_obs[5]
        roll, pitch, yaw = raw_obs[6], raw_obs[7], raw_obs[8]

        cy, sy = math.cos(yaw), math.sin(yaw)
        dx_body =  cy * dx + sy * dy
        dy_body = -sy * dx + cy * dy

        return np.array([
            dx_body, dy_body, dz,
            vx, vy, vz,
            roll, pitch,
            math.sin(yaw), math.cos(yaw),
        ], dtype=np.float32)

    def _noisy_obs(self) -> np.ndarray:
        """Observation returned to the policy — adds Gaussian sensor noise (on
        the raw world-frame state) if enabled, then — in randomized mode only
        — converts to the body-frame/sin-cos-yaw policy obs. In legacy mode
        (randomize=False) the raw 9-dim world-frame/raw-yaw array is returned
        as-is, byte-identical to the original observation contract that
        quadrotor_hover_ppo.zip was trained against. Reward/crash logic
        always reads the clean self._obs, never this."""
        raw = self._obs.copy()
        if self.obs_noise:
            raw[0:3] += self.np_random.normal(0.0, self.noise_std_pos, size=3)
            raw[3:6] += self.np_random.normal(0.0, self.noise_std_vel, size=3)
            raw[6:9] += self.np_random.normal(0.0, self.noise_std_angle, size=3)
        obs = self._to_policy_obs(raw) if self.randomize else raw
        if self.obs_noise:
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1

        # Heartbeat every 10 steps (~5Hz) to keep controller armed
        if self.current_step % 10 == 0:
            enable_msg = Bool()
            enable_msg.data = True
            self._enable_pub.publish(enable_msg)

        # Linear scale of 0.35 m/s in randomized mode (up from the original
        # 0.1, sized for the old fixed-point hover task) so the drone can
        # physically cover a randomized target up to ~2.5m away within a
        # 500-step/25s episode with margin to spare (worst case ~7.3s at full
        # speed, leaving ~18s for correction). At 0.1 m/s the theoretical max
        # reach was 2.5m with ZERO margin, which left PPO with no learnable
        # reward gradient on far targets. Legacy mode keeps the original 0.1
        # scale — quadrotor_hover_ppo.zip was trained against it, and a
        # bigger scale would make it fly faster/more aggressively than it
        # ever learned to handle.
        linear_scale = 0.35 if self.randomize else 0.1
        cmd = Twist()
        cmd.linear.x  = float(np.clip(action[0], -1.0, 1.0)) * linear_scale
        cmd.linear.y  = float(np.clip(action[1], -1.0, 1.0)) * linear_scale
        cmd.linear.z  = float(np.clip(action[2], -1.0, 1.0)) * linear_scale
        cmd.angular.z = float(np.clip(action[3], -1.0, 1.0)) * 0.1
        self._cmd_pub.publish(cmd)

        if self.control_dt is None:
            # Legacy pacing: one executor spin, returns on the FIRST callback
            # (IMU at 100Hz beats odom at 50Hz), so a step spans ~10-15ms of
            # sim time. quadrotor_hover_ppo.zip was trained against this
            # timing, so it must stay byte-identical in legacy mode.
            self._spin_once(timeout_sec=0.05)
        else:
            # Sim-time pacing: spin until the odometry stamp has advanced a
            # full control period. Exact in sim time at any real_time_factor,
            # and the observation is guaranteed fresh. Wall-clock guard in
            # case physics is paused/stalled (mirrors _wait_for_obs).
            # The 1e-6 epsilon keeps a control_dt that is an exact multiple
            # of the odom period (50Hz -> 0.02s) from flakily waiting one
            # extra sample on float rounding of the stamp arithmetic.
            deadline = self._odom_stamp + self.control_dt - 1e-6
            wall_start = time.time()
            while self._odom_stamp < deadline:
                self._spin_once(timeout_sec=0.05)
                if time.time() - wall_start > 2.0:
                    logger.warning("timed out waiting for control_dt of sim time!")
                    break

        reward, crashed = self._compute_reward()
        dist = self._current_dist()
        success = self._update_success_hold(crashed, dist)

        truncated  = self.current_step >= self.max_steps
        terminated = crashed or success

        info = {"dist": dist}
        if terminated or truncated:
            # Episode outcome: "won" via success-hold, or (when success
            # termination is off) ended inside the success radius. Feeds the
            # curriculum and SB3's success-rate logging convention.
            info["is_success"] = bool(success or (not crashed and dist < self.success_threshold))
            self._curriculum_record(info["is_success"])

        return self._noisy_obs(), float(reward), terminated, truncated, info

    def close(self):
        self._gz_node = None

        if hasattr(self, "_executor") and self._executor is not None:
            try:
                self._executor.remove_node(self.node)
            except Exception:
                pass
            try:
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None

        if hasattr(self, "node") and self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None

        if hasattr(self, "_context") and self._context is not None:
            try:
                rclpy.shutdown(context=self._context)
            except Exception:
                pass
            self._context = None

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        dx, dy, dz = self._obs[0], self._obs[1], self._obs[2]
        vz = self._obs[5]
        roll, pitch = self._obs[6], self._obs[7]

        abs_x = dx + self.target_xyz[0]
        abs_y = dy + self.target_xyz[1]
        abs_z = dz + self.target_xyz[2]

        # Crash conditions use ABSOLUTE arena bounds — independent of target_xyz,
        # since the target is randomized per-episode but the arena's physical
        # limits (floor, ceiling, walls) are not.
        crashed = (
            abs_z > self.max_z or
            abs_z < self.min_z or
            abs(abs_x) > self.max_xy or
            abs(abs_y) > self.max_xy
        )

        if crashed:
            return self.crash_penalty, True

        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if self.stabilization_gate_sigma > 0:
            stabilization_gate = math.exp(-(dist ** 2) / (2 * self.stabilization_gate_sigma ** 2))
        else:
            stabilization_gate = 1.0  # no gating — full penalty always (legacy behavior)

        reward  =  1.0                                              # survival bonus
        reward -= 1.0 * dist                                        # 3D distance-to-target penalty
        reward -= 0.2 * stabilization_gate * abs(vz)                # vertical velocity damping (gated)
        reward -= 0.05 * stabilization_gate * (abs(roll) + abs(pitch))  # attitude stabilization (gated)
        # Coefficients are a starting point — tune after the first training run.

        if self.progress_coef and self._prev_dist is not None:
            # Potential-based shaping: credits THIS step for the distance it
            # actually closed, rather than leaving credit assignment purely
            # to TD bootstrapping on the static distance penalty above.
            reward += self.progress_coef * (self._prev_dist - dist)
        self._prev_dist = dist

        if self.precision_bonus:
            # Sharply peaked at dist=0, negligible beyond a few sigma — gives
            # a "final approach" incentive the linear distance term can't,
            # without disturbing coarse long-range navigation.
            reward += self.precision_bonus * math.exp(-(dist ** 2) / (2 * self.precision_sigma ** 2))

        if self.success_bonus and dist < self.success_threshold:
            reward += self.success_bonus

        return reward, False

    # ------------------------------------------------------------------
    # Gazebo reset helpers
    # ------------------------------------------------------------------

    def _pause_physics(self):
        req = WorldControl()
        req.pause = True
        self._gz_node.request('/world/empty/control', req, WorldControl, Boolean, 5000)

    def _unpause_physics(self):
        req = WorldControl()
        req.pause = False
        self._gz_node.request('/world/empty/control', req, WorldControl, Boolean, 5000)
        spin_deadline = time.time() + 0.5
        while time.time() < spin_deadline:
            self._spin_once(timeout_sec=0.05)

    def _reset_pose_only(self, x, y, z, roll, pitch, yaw):
        """Teleport drone to (x, y, z, roll, pitch, yaw) without deleting/respawning."""

        self._odom_ready = False
        self._imu_ready = False

        self._pause_physics()
        time.sleep(0.2)

        qx, qy, qz, qw = self._euler_to_quat(roll, pitch, yaw)
        req = Pose()
        req.name = "quadrotor"
        req.position.x = x
        req.position.y = y
        req.position.z = z
        req.orientation.w = qw
        req.orientation.x = qx
        req.orientation.y = qy
        req.orientation.z = qz
        self._gz_node.request('/world/empty/set_pose', req, Pose, Boolean, 5000)

        self._last_commanded_pose = (x, y, z, yaw)
        time.sleep(0.5)
        self._unpause_physics()

    def _reset_gazebo(self):
        """Full delete and respawn — only used if pose reset is insufficient."""

        subprocess.run([
            'gz', 'service', '-s', '/world/empty/control',
            '--reqtype', 'gz.msgs.WorldControl',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', 'pause: true'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.run([
            'gz', 'service', '-s', '/world/empty/remove',
            '--reqtype', 'gz.msgs.Entity',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', 'name: "quadrotor" type: MODEL'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)

        sdf_file = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', 'models', 'quadrotor', 'quadrotor.sdf'
        ))
        result = subprocess.run([
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-name', 'quadrotor',
            '-file', sdf_file,
            '-x', '0', '-y', '0', '-z', '1.0'
        ], capture_output=True, text=True)

        if 'Entity creation successful' not in result.stdout and \
           'Entity creation successful' not in result.stderr:
            print("[env] WARNING: drone respawn may have failed!")
            print(f"[env] stdout: {result.stdout.strip()}")
            print(f"[env] stderr: {result.stderr.strip()}")
        else:
            print("[env] Drone respawned successfully.")

        time.sleep(1.0)

        subprocess.run([
            'gz', 'service', '-s', '/world/empty/control',
            '--reqtype', 'gz.msgs.WorldControl',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', 'pause: false'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _at_spawn_pose(self) -> bool:
        """Odom looks like a successful teleport to the most recently commanded pose."""
        cx, cy, cz, _ = self._last_commanded_pose
        abs_x = self._obs[0] + self.target_xyz[0]
        abs_y = self._obs[1] + self.target_xyz[1]
        abs_z = self._obs[2] + self.target_xyz[2]
        return (
            abs(abs_x - cx) < 0.5
            and abs(abs_y - cy) < 0.5
            and abs(abs_z - cz) < 0.35
        )

    def _wait_for_obs(self, require_spawn_pose: bool = False):
        """Spin until odom + imu arrived, then optionally give the drone a
        short, bounded window to settle near the teleport pose.

        Two phases with very different failure semantics:
          1. Sensor arrival — if odom/imu never show up the bridge or sim is
             broken; that is worth a real WARNING (+ debug dump).
          2. Spawn-pose settle — after a teleport the re-armed controller only
             holds zero velocity, so nothing actively returns the drone to the
             exact commanded height and it often settles a little off. That is
             benign (training randomizes spawn anyway), so wait only briefly,
             then proceed and log at DEBUG instead of flooding WARNINGs.
        """
        self._odom_ready = False
        self._imu_ready = False

        # Phase 1: sensors must actually arrive.
        deadline = time.time() + self._obs_wait_timeout
        while time.time() < deadline:
            self._spin_once(timeout_sec=0.05)
            if self._odom_ready and self._imu_ready:
                break
        else:
            logger.warning("timed out waiting for observations!")
            try:
                logger.warning(
                    "obs wait debug: odom_ready=%s imu_ready=%s spawn_pose=%s "
                    "obs=(dx=%.2f, dy=%.2f, dz=%.2f) target=%s commanded=%s",
                    self._odom_ready,
                    self._imu_ready,
                    self._at_spawn_pose(),
                    self._obs[0],
                    self._obs[1],
                    self._obs[2],
                    self.target_xyz,
                    self._last_commanded_pose,
                )
                imu_pubs = self.node.count_publishers('/quadrotor/imu')
                odom_pubs = self.node.count_publishers('/quadrotor/odom')
                logger.warning(
                    "obs wait debug: ros publishers imu=%s odom=%s",
                    imu_pubs,
                    odom_pubs,
                )
            except Exception as e:
                logger.error("obs wait debug: failed to query topics/publishers: %r", e)
            return

        # Phase 2: brief, bounded settle near the commanded spawn pose.
        if require_spawn_pose:
            settle_deadline = time.time() + self._spawn_settle_timeout
            while time.time() < settle_deadline:
                self._spin_once(timeout_sec=0.05)
                if self._at_spawn_pose():
                    return
            logger.debug(
                "spawn pose not reached within %.1fs (obs dz=%.2f, commanded=%s); "
                "proceeding with settled pose",
                self._spawn_settle_timeout,
                self._obs[2],
                self._last_commanded_pose,
            )

    @staticmethod
    def _quat_to_euler(x, y, z, w):
        roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
        yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw

    @staticmethod
    def _euler_to_quat(roll, pitch, yaw):
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        return qx, qy, qz, qw