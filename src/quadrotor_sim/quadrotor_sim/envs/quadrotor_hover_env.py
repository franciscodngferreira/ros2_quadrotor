import gymnasium as gym
import numpy as np
import os
import rclpy
import rclpy.executors
import rclpy.context
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
import subprocess
import time
import math


class QuadrotorHoverEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500):
        super().__init__()

        self.target_z = 1.0
        self.max_steps = max_steps
        self.current_step = 0
        self.min_z = 0.05
        self.max_z = 3.0
        self.max_xy = 3.0

        print(
            f"[env] max_steps={self.max_steps} "
            f"bounds: min_z={self.min_z} max_z={self.max_z} max_xy={self.max_xy}"
        )

        self._first_reset = True

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space wider than crash bounds so agent sees gradient near boundary
        obs_high = np.array([
            5.0, 5.0, 5.0,          # x, y, z_error (z - target_z)
            5.0, 5.0, 5.0,          # vx, vy, vz
            math.pi, math.pi/2, math.pi  # roll, pitch, yaw
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        self._obs = np.zeros(9, dtype=np.float32)
        self._odom_received = False
        self._imu_received = False
        self._printed_first_odom = False
        self._printed_first_imu = False
        self._debug_rx = os.environ.get("QUADROTOR_ENV_DEBUG_RX", "").lower() in {"1", "true", "yes", "on"}

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

        self._cmd_pub = self.node.create_publisher(
            Twist, '/quadrotor/cmd_vel', 10)
        self._enable_pub = self.node.create_publisher(
            Bool, '/quadrotor/enable', 10)

        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._odom_sub = self.node.create_subscription(
            Odometry, '/quadrotor/odom', self._odom_cb, qos_reliable)
        self._imu_sub = self.node.create_subscription(
            Imu, '/quadrotor/imu', self._imu_cb, qos_reliable)

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
        subprocess.run([
            'gz', 'service', '-s', '/world/empty/control',
            '--reqtype', 'gz.msgs.WorldControl',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', 'pause: false'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self._wait_for_obs()
        print("[env] Controller enabled and physics running.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg):
        if self._debug_rx and (not self._printed_first_odom) and (not self._odom_received):
            print("[env] First odom received")
            self._printed_first_odom = True
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self._obs[0] = pos.x
        self._obs[1] = pos.y
        self._obs[2] = pos.z - self.target_z   # z error, not raw z
        self._obs[3] = vel.x
        self._obs[4] = vel.y
        self._obs[5] = vel.z
        self._odom_received = True

    def _imu_cb(self, msg):
        if self._debug_rx and (not self._printed_first_imu) and (not self._imu_received):
            print("[env] First imu received")
            self._printed_first_imu = True
        q = msg.orientation
        roll, pitch, yaw = self._quat_to_euler(q.x, q.y, q.z, q.w)
        self._obs[6] = roll
        self._obs[7] = pitch
        self._obs[8] = yaw
        self._imu_received = True

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if self._first_reset:
            self._first_reset = False
            # Re-arm controller — obs already populated from __init__
            enable_msg = Bool()
            enable_msg.data = True
            start = time.time()
            while time.time() - start < 0.5:
                self._enable_pub.publish(enable_msg)
                self._spin_once(timeout_sec=0.05)
            return self._obs.copy(), {}

        # Subsequent resets — teleport drone back, don't delete/respawn
        self._odom_received = False
        self._imu_received = False
        self._printed_first_odom = False
        self._printed_first_imu = False

        self._reset_pose_only()

        # Re-arm controller after teleport
        enable_msg = Bool()
        enable_msg.data = True
        start = time.time()
        while time.time() - start < 1.0:
            self._enable_pub.publish(enable_msg)
            self._cmd_pub.publish(Twist())
            self._spin_once(timeout_sec=0.05)

        self._wait_for_obs()
        return self._obs.copy(), {}

    def step(self, action):
        self.current_step += 1

        # Heartbeat every 10 steps (~5Hz) to keep controller armed
        if self.current_step % 10 == 0:
            enable_msg = Bool()
            enable_msg.data = True
            self._enable_pub.publish(enable_msg)

        cmd = Twist()
        cmd.linear.x  = float(np.clip(action[0], -1.0, 1.0)) * 0.1
        cmd.linear.y  = float(np.clip(action[1], -1.0, 1.0)) * 0.1
        cmd.linear.z  = float(np.clip(action[2], -1.0, 1.0)) * 0.1
        cmd.angular.z = float(np.clip(action[3], -1.0, 1.0)) * 0.1
        self._cmd_pub.publish(cmd)

        self._spin_once(timeout_sec=0.05)

        reward, crashed = self._compute_reward()

        truncated  = self.current_step >= self.max_steps
        terminated = crashed

        return self._obs.copy(), float(reward), terminated, truncated, {}

    def close(self):
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
        x, y  = self._obs[0], self._obs[1]
        z_err = self._obs[2]   # z - target_z
        vz    = self._obs[5]

        # Crash conditions
        crashed = (
            z_err > (self.max_z - self.target_z) or      # too high
            z_err < -(self.target_z - self.min_z) or     # too low / floor
            abs(x) > self.max_xy or
            abs(y) > self.max_xy
        )

        if crashed:
            return -1000.0, True

        reward  =  1.0                      # survival bonus
        reward -= 2.0 * abs(z_err)          # height error penalty
        reward -= 0.3 * (abs(x) + abs(y))  # xy drift penalty
        reward -= 0.3 * abs(vz)            # vertical velocity penalty

        return reward, False

    # ------------------------------------------------------------------
    # Gazebo reset helpers
    # ------------------------------------------------------------------

    def _reset_pose_only(self):
        """Teleport drone back to start without deleting/respawning.
        Much faster and more reliable than delete+respawn."""

        # Pause physics
        subprocess.run([
            'gz', 'service', '-s', '/world/empty/control',
            '--reqtype', 'gz.msgs.WorldControl',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', 'pause: true'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(0.2)

        # Teleport drone back to z=1.0 with identity orientation
        subprocess.run([
            'gz', 'service', '-s', '/world/empty/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', 'name: "quadrotor" position: {x: 0 y: 0 z: 1.0} orientation: {w: 1 x: 0 y: 0 z: 0}'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait while paused so rotor state and velocity decay
        time.sleep(0.5)

        # Unpause physics
        subprocess.run([
            'gz', 'service', '-s', '/world/empty/control',
            '--reqtype', 'gz.msgs.WorldControl',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', 'pause: false'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(0.3)

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

    def _wait_for_obs(self):
        """Spin until we have at least one odom + imu message."""
        timeout = 10.0
        start = time.time()
        printed_debug = False
        while not (self._odom_received and self._imu_received):
            self._executor.spin_once(timeout_sec=0.0)   # drain queued callbacks
            self._executor.spin_once(timeout_sec=0.1)   # wait for next message
            if time.time() - start > timeout:
                print("[env] WARNING: timed out waiting for observations!")
                if not printed_debug:
                    printed_debug = True
                    try:
                        imu_pubs  = self.node.count_publishers('/quadrotor/imu')
                        odom_pubs = self.node.count_publishers('/quadrotor/odom')
                        print(f"[env] Debug: publishers imu={imu_pubs} odom={odom_pubs}")
                        topics = self.node.get_topic_names_and_types()
                        interesting = [
                            (name, types) for (name, types) in topics
                            if name.startswith('/quadrotor/')
                        ]
                        print(f"[env] Debug: seen {len(interesting)} /quadrotor/* topics")
                        for name, types in sorted(interesting):
                            print(f"[env]   {name}: {types}")
                    except Exception as e:
                        print(f"[env] Debug: failed to query topics/publishers: {e!r}")
                break

    @staticmethod
    def _quat_to_euler(x, y, z, w):
        roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
        yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw