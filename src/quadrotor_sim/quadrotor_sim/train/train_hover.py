import subprocess
import time
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def _prefer_source_tree():
    """
    Ensure we import quadrotor_sim from the workspace source tree, not a stale
    installed copy under install/.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(this_dir, ".."))  # .../quadrotor_sim/
    src_root = os.path.abspath(os.path.join(pkg_root, ".."))  # .../src/quadrotor_sim/
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


_prefer_source_tree()

from quadrotor_sim.envs.quadrotor_hover_env import QuadrotorHoverEnv  # noqa: E402


def _ensure_runtime_dirs():
    """
    Keep ROS 2 / Gazebo from writing into ~/.ros and ~/.gz.
    This matters in sandboxed / containerized environments and is harmless otherwise.
    """
    ws = os.path.abspath(os.getcwd())
    ros_home = os.environ.setdefault("ROS_HOME", os.path.join(ws, ".ros"))
    ros_log_dir = os.environ.setdefault("ROS_LOG_DIR", os.path.join(ws, "log", "ros"))
    gz_home = os.environ.setdefault("GZ_HOME", os.path.join(ws, ".gz"))
    os.makedirs(ros_home, exist_ok=True)
    os.makedirs(ros_log_dir, exist_ok=True)
    os.makedirs(gz_home, exist_ok=True)


def launch_gazebo():
    print("[train] Cleaning up any existing Gazebo instances...")
    subprocess.run(['pkill', '-f', 'gz sim'], capture_output=True)
    subprocess.run(['pkill', '-f', 'ros2 launch'], capture_output=True)
    subprocess.run(['pkill', '-f', 'parameter_bridge'], capture_output=True)
    time.sleep(2.0)

    pkg_dir = subprocess.check_output(
        ['ros2', 'pkg', 'prefix', 'quadrotor_sim']
    ).decode().strip()
    world_file = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'worlds', 'empty.sdf'
    ))
    sdf_file = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'models', 'quadrotor', 'quadrotor.sdf'
    ))

    print("[train] Launching Gazebo...")
    gz_proc = subprocess.Popen(
        ['gz', 'sim', '-s', world_file],
        env=os.environ.copy(),
        stderr=subprocess.DEVNULL,
    )

    print("[train] Waiting for Gazebo to be ready...")
    while True:
        result = subprocess.run(
            ['gz', 'topic', '-l'],
            capture_output=True, text=True
        )
        if '/world/empty/clock' in result.stdout:
            print("[train] Gazebo ready.")
            break
        time.sleep(1.0)

    time.sleep(3.0)

    print("[train] Launching bridge...")
    bridge_proc = subprocess.Popen([
        'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        '/quadrotor/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/quadrotor/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/quadrotor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        '/quadrotor/enable@std_msgs/msg/Bool]gz.msgs.Boolean',
    ])
    time.sleep(2.0)

    print("[train] Pausing physics...")
    subprocess.run([
        'gz', 'service', '-s', '/world/empty/control',
        '--reqtype', 'gz.msgs.WorldControl',
        '--reptype', 'gz.msgs.Boolean',
        '--timeout', '5000',
        '--req', 'pause: true'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[train] Spawning drone at z=1.0...")
    subprocess.run([
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-name', 'quadrotor',
        '-file', sdf_file,
        '-x', '0', '-y', '0', '-z', '1.0'
    ])

    return gz_proc, bridge_proc


def main():
    _ensure_runtime_dirs()
    # 1. Launch Gazebo headless
    gz_proc, bridge_proc = launch_gazebo()

    try:
        # 2. Create environment
        print("[train] Creating environment...")
        env = QuadrotorHoverEnv()

        # 3. Sanity check
        #print("[train] Checking environment...")
        #check_env(env, warn=True)

        # 4. Create PPO agent
        print("[train] Creating PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log="./hover_tensorboard/"
        )

        # 5. Train
        print("[train] Starting training...")
        model.learn(total_timesteps=100_000)

        # 6. Save
        model.save("quadrotor_hover_ppo")
        print("[train] Model saved to quadrotor_hover_ppo.zip")

    finally:
        print("[train] Shutting down...")
        gz_proc.terminate()
        bridge_proc.terminate()
        gz_proc.wait()
        bridge_proc.wait()


if __name__ == "__main__":
    main()
