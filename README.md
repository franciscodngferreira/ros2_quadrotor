# quadrotor_sim (ROS 2 + Gazebo Sim + RL)

This workspace contains a ROS 2 package (`quadrotor_sim`) that launches a simple quadrotor model in **Gazebo Sim (gz-sim)**, bridges a small set of topics to **ROS 2**, and provides a **Gymnasium** environment used to train a hover policy with **Stable-Baselines3 (PPO)**.

## Repository layout

- `src/`
  - `quadrotor_sim/` (ROS 2 package)
    - `package.xml`: ROS dependencies (`rclpy`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `ros_gz_sim`, `ros_gz_bridge`, …)
    - `CMakeLists.txt`: installs `models/`, `worlds/`, `launch/` + installs the Python package
    - `launch/quadrotor.launch.py`: starts Gazebo Sim and the `ros_gz_bridge/parameter_bridge`
    - `models/quadrotor/quadrotor.sdf`: quadrotor model + multicopter control + sensors / odom publisher
    - `worlds/empty.sdf`: minimal world used by the launch file
    - `quadrotor_sim/` (Python package)
      - `envs/quadrotor_hover_env.py`: Gymnasium environment that:
        - publishes `/quadrotor/cmd_vel` (`geometry_msgs/Twist`)
        - publishes `/quadrotor/enable` (`std_msgs/Bool`) to arm/enable the controller
        - subscribes to `/quadrotor/imu` (`sensor_msgs/Imu`)
        - subscribes to `/quadrotor/odom` (`nav_msgs/Odometry`)
      - `train/train_hover.py`: launches Gazebo headless and trains PPO

## ROS/Gazebo topic interface (bridged)

The package uses `ros_gz_bridge` to bridge these topics:

- **ROS → Gazebo**
  - `/quadrotor/cmd_vel` (`geometry_msgs/msg/Twist`)
  - `/quadrotor/enable` (`std_msgs/msg/Bool`)
- **Gazebo → ROS**
  - `/quadrotor/imu` (`sensor_msgs/msg/Imu`)
  - `/quadrotor/odom` (`nav_msgs/msg/Odometry`)

## Build

From the workspace root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select quadrotor_sim --symlink-install
source install/setup.bash
```

## Run the simulator (headless)

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch quadrotor_sim quadrotor.launch.py
```

To verify data is flowing:

```bash
ros2 topic echo --once /quadrotor/imu
ros2 topic echo --once /quadrotor/odom
```

## Train the hover policy (PPO)

This script launches Gazebo headless and runs a short PPO training loop:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# activate your RL venv that contains stable-baselines3, gymnasium, etc.
source ~/rl_venv/bin/activate

export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/quadrotor_sim
PYTHONUNBUFFERED=1 python3 src/quadrotor_sim/quadrotor_sim/train/train_hover.py
```

Outputs:

- TensorBoard logs: `hover_tensorboard/`
- Saved model: `quadrotor_hover_ppo.zip` (written in the working directory)

## Debug tips

- **Too much env logging**: the env has optional receive-debug prints gated by:
  - `QUADROTOR_ENV_DEBUG_RX=1` (enable)
  - unset / `0` (disable)
- **ROS logging permission issues**: if your environment can’t write `~/.ros`, set:

```bash
export ROS_HOME=$PWD/.ros
export ROS_LOG_DIR=$PWD/log/ros
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"
```

## Notes / known limitations

- The environment’s reset defaults to a “soft reset” (does not hard-reset Gazebo) because hard resets via Gazebo CLI services can stall sensor streams in some setups.
- The package currently has placeholder metadata in `package.xml` (version/license/maintainer).

