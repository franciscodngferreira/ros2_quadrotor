import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    pkg_dir = get_package_share_directory('quadrotor_sim')
    world_file = os.path.join(pkg_dir, 'worlds', 'empty.sdf')
    sdf_file = os.path.join(pkg_dir, 'models', 'quadrotor', 'quadrotor.sdf')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={'gz_args': f'-s -r {world_file}'}.items()
    )

    spawn_drone = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'quadrotor',
            '-file', sdf_file,
            '-x', '0', '-y', '0', '-z', '1.0',
        ],
        output='screen'
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
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_drone,
        bridge,
    ])
