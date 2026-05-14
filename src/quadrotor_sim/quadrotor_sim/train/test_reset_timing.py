#!/usr/bin/env python3
"""
Timing diagnostic for Gazebo reset sequence.
Run while Gazebo is already up (use train_hover.py to launch it first,
then Ctrl+C before training starts, or add a breakpoint).
Or launch Gazebo separately and run this standalone.
"""
import subprocess
import time
import rclpy
import rclpy.context
import rclpy.executors
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

odom_received = False
odom_z = None

def odom_cb(msg):
    global odom_received, odom_z
    odom_received = True
    odom_z = msg.pose.pose.position.z

def gz_control(req):
    subprocess.run([
        'gz', 'service', '-s', '/world/empty/control',
        '--reqtype', 'gz.msgs.WorldControl',
        '--reptype', 'gz.msgs.Boolean',
        '--timeout', '2000',
        '--req', req
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def wait_for_odom(executor, timeout=10.0):
    global odom_received
    odom_received = False
    start = time.time()
    while not odom_received:
        executor.spin_once(timeout_sec=0.1)
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"  TIMEOUT after {timeout}s — odom never arrived")
            return False
    print(f"  odom received in {time.time()-start:.2f}s, z={odom_z:.3f}")
    return True

def main():
    ctx = rclpy.context.Context()
    rclpy.init(context=ctx)
    node = rclpy.create_node('timing_test', context=ctx)
    executor = rclpy.executors.SingleThreadedExecutor(context=ctx)
    executor.add_node(node)

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    node.create_subscription(Odometry, '/quadrotor/odom', odom_cb, qos)
    enable_pub = node.create_publisher(Bool, '/quadrotor/enable', 10)
    cmd_pub = node.create_publisher(Twist, '/quadrotor/cmd_vel', 10)

    enable_msg = Bool(); enable_msg.data = True
    cmd = Twist()

    for sleep_after_unpause in [0.5, 1.0, 2.0]:
        print(f"\n--- Testing pause-reset-unpause-enable, sleep={sleep_after_unpause}s ---")

        gz_control('pause: true')
        gz_control('reset: {all: true}')
        gz_control('pause: false')
        
        print(f"  sleeping {sleep_after_unpause}s...")
        time.sleep(sleep_after_unpause)

        # Enable after unpause
        start = time.time()
        while time.time() - start < 2.0:
            enable_pub.publish(enable_msg)
            cmd_pub.publish(cmd)
            executor.spin_once(timeout_sec=0.05)

        print("  waiting for odom...")
        wait_for_odom(executor)

    rclpy.shutdown(context=ctx)

if __name__ == '__main__':
    main()