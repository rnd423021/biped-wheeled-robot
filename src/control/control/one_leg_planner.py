import enum
import numpy as np
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String
from geometry_msgs.msg import Point

from axel_planner.biped_wheeled_leg import BipedWheeledLeg
from axel_planner.planner import get_trajectory


class TrajectoryExecutorStatus(enum.Enum):
    IDLE = 0
    RUNNING_TRAJ = 1
    PAUSED = 2


class TrajectoryExecutor:
    def __init__(self):

        self.leg_model = BipedWheeledLeg()
        self.status = TrajectoryExecutorStatus.IDLE

        # Trajectory data
        self.xy_traj = None
        self.time_traj = None
        self.num_points = 0
        self.traj_dt = 0.01
        self.q_values_traj = None
        self.start_time_traj = None
        self.current_commanded_q = None

        # Error tracking
        self.error_log = []
        self.time_log = []
        self.desired_q_log = []
        self.actual_q_log = []

    def set_trajectory(self, xy_traj, time_traj, num_points, traj_dt):

        self.xy_traj = xy_traj
        self.time_traj = time_traj
        self.num_points = num_points
        self.traj_dt = traj_dt
        self.q_values_traj = np.array(
            [self.leg_model.ik_solve(xy[0], xy[1]) for xy in self.xy_traj]
        )
        self.start_time_traj = None
        self.current_commanded_q = None
        self.status = TrajectoryExecutorStatus.IDLE

    def start(self):
        """Start trajectory execution"""
        if self.q_values_traj is not None:
            self.start_time_traj = None
            self.status = TrajectoryExecutorStatus.RUNNING_TRAJ

    def pause(self):
        """Pause trajectory execution"""

        self.status = TrajectoryExecutorStatus.PAUSED

    def stop(self):
        """Stop trajectory execution"""

        self.start_time_traj = None
        self.status = TrajectoryExecutorStatus.IDLE

    def get_error_data(self):
        """Get error tracking data"""

        return {
            "error": np.array(self.error_log) if self.error_log else np.array([]),
            "time": np.array(self.time_log) if self.time_log else np.array([]),
            "desired_q": (
                np.array(self.desired_q_log) if self.desired_q_log else np.array([])
            ),
            "actual_q": (
                np.array(self.actual_q_log) if self.actual_q_log else np.array([])
            ),
        }


class TrajectoryExecutorNode(Node):
    """ROS2 Node for trajectory execution using t_motor_node"""

    def __init__(self):
        super().__init__("trajectory_executor_node")

        # Initialize trajectory traj_generator
        self.traj_generator = TrajectoryExecutor()
        self.leg_model = BipedWheeledLeg()

        # Current actual joint positions from motors
        self.actual_joint_pos = np.array([0.0, 0.0])  # [hip, knee]

        # Publishers
        self.status_pub = self.create_publisher(String, "trajectory_status", 10)
        self.desired_pos_pub = self.create_publisher(
            Float32MultiArray, "t_motor/desired_position", 10
        )
        self.end_effector_pub = self.create_publisher(Point, "end_effector_pos", 10)

        # Subscribers
        self.create_subscription(
            Float32MultiArray,
            "t_motor/actual_position",
            self.actual_position_callback,
            10,
        )
        self.create_subscription(
            Point, "target_position", self.target_position_callback, 10
        )
        self.create_subscription(String, "command", self.command_callback, 10)

        # Timer for trajectory execution (uses traj_dt from traj_generator)
        self.traj_timer = self.create_timer(0.01, self.trajectory_callback)

        # Status publishing timer (10 Hz)
        self.create_timer(0.1, self.publish_status)

        self.get_logger().info("Trajectory Executor Node initialized")

    def actual_position_callback(self, msg: Float32MultiArray):
        """Update actual joint positions from t_motor_node"""
        self.actual_joint_pos = np.array(msg.data)

    def target_position_callback(self, msg: Point):
        """Handle new target position"""
        target_xy = np.array([msg.x, msg.y])

        # Get current position from actual joint positions
        initial_x, initial_y = self.leg_model.forward_kinematics(
            self.actual_joint_pos[0], self.actual_joint_pos[1]
        )

        # Generate trajectory
        x_traj, y_traj, time_traj, num_points, traj_dt = get_trajectory(
            [initial_x, initial_y],
            target_xy,
            traj_velocity=0.3,
            traj_point_per_meter=600,
        )

        # Set trajectory
        self.traj_generator.set_trajectory(
            xy_traj=np.array([x_traj, y_traj]).T,
            time_traj=time_traj,
            num_points=num_points,
            traj_dt=traj_dt,
        )

        self.traj_generator.start()
        
        self.get_logger().info(f"Trajectory set to target: ({msg.x:.3f}, {msg.y:.3f})")

    def command_callback(self, msg: String):
        """Handle trajectory control commands"""
        command = msg.data.lower()

        if command == "start":
            self.traj_generator.start()
            self.get_logger().info("Trajectory started")
        elif command == "pause":
            self.traj_generator.pause()
            self.get_logger().info("Trajectory paused")
        elif command == "stop":
            self.traj_generator.stop()
            self.get_logger().info("Trajectory stopped")
        else:
            self.get_logger().warn(f"Unknown command: {command}")

    def publish_status(self):
        """Publish current status and end effector position"""
        # Publish status
        status_msg = String()
        status_msg.data = self.traj_generator.status.name
        self.status_pub.publish(status_msg)

        # Publish end effector position
        ee_x, ee_y = self.leg_model.forward_kinematics(
            self.actual_joint_pos[1], self.actual_joint_pos[0]
        )
        ee_msg = Point()
        ee_msg.x = float(ee_x)
        ee_msg.y = float(ee_y)
        ee_msg.z = 0.0
        self.end_effector_pub.publish(ee_msg)

    def trajectory_callback(self):
        """Timer callback for trajectory execution"""
        status = self.traj_generator.status

        # Skip if not running
        if status != TrajectoryExecutorStatus.RUNNING_TRAJ:
            return

        # Initialize trajectory start time on first iteration
        if self.traj_generator.start_time_traj is None:
            self.traj_generator.start_time_traj = time.perf_counter()

        # Calculate current position in trajectory
        elapsed_time = time.perf_counter() - self.traj_generator.start_time_traj
        traj_index = min(
            int(elapsed_time / self.traj_generator.traj_dt), self.traj_generator.num_points - 1
        )

        # Send commands to t_motor_node
        current_q = self.traj_generator.q_values_traj[traj_index]

        desired_pos_msg = Float32MultiArray()
        desired_pos_msg.data = [float(current_q[0]), float(current_q[1])]
        self.desired_pos_pub.publish(desired_pos_msg)

        # Store current commanded position for logging
        self.traj_generator.current_commanded_q = current_q
 

        # Stop when trajectory is complete
        if traj_index >= self.traj_generator.num_points - 1:
            self.traj_generator.start_time_traj = None
            self.traj_generator.status = TrajectoryExecutorStatus.IDLE
            self.get_logger().info("Trajectory completed")

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        self.get_logger().info("Shutting down trajectory traj_generator")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryExecutorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
