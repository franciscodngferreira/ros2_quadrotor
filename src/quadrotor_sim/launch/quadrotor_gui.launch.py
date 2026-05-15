import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    """Gazebo Sim with GUI: headless server (-s) + GUI client (-g)."""

    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_dir = get_package_share_directory('quadrotor_sim')
    world_file = os.path.join(pkg_dir, 'worlds', 'empty.sdf')
    sdf_file = os.path.join(pkg_dir, 'models', 'quadrotor', 'quadrotor.sdf')

    # Physics server (no GUI). Same pattern as turtlebot3_gazebo / nav2_minimal_tb4_sim.
    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={
            'gz_args': f'-r -s -v2 {world_file}',
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # GUI client connects to the running server.
    gz_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={'gz_args': '-g -v2 '}.items(),
    )

    spawn_drone = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', 'empty',
            '-name', 'quadrotor',
            '-file', sdf_file,
            '-x', '0', '-y', '0', '-z', '1.0',
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/quadrotor/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/quadrotor/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/quadrotor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/quadrotor/enable@std_msgs/msg/Bool]gz.msgs.Boolean',
        ],
        output='screen',
    )

    return LaunchDescription([
        gz_server,
        gz_gui,
        bridge,
        # Wait for /world/empty before spawning (GUI client can start in parallel).
        TimerAction(period=5.0, actions=[spawn_drone]),
    ])
