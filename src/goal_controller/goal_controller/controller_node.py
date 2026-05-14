import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import math

class GoalController(Node):
    def __init__(self):
        super().__init__('goal_controller')

        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 2.0)

        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value

        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.get_logger().info(f'Goal controller started. Target: ({self.goal_x}, {self.goal_y})')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

    def control_loop(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < 0.1:
            self.get_logger().info('Goal reached!', throttle_duration_sec=1.0)
            self.publisher.publish(cmd)
            return

        angle_to_goal = math.atan2(dy, dx)
        angle_error = angle_to_goal - self.current_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        Kp_linear = 0.4
        Kp_angular = 1.0

        cmd.twist.linear.x = Kp_linear * distance
        cmd.twist.angular.z = Kp_angular * angle_error

        cmd.twist.linear.x = min(cmd.twist.linear.x, 0.5)
        cmd.twist.angular.z = max(min(cmd.twist.angular.z, 1.0), -1.0)

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = GoalController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()