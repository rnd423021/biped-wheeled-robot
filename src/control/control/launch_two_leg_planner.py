from control.one_leg_planner import (
    TrajectoryExecutorNode,
    TrajectoryExecutorSide,
)
import rclpy
from rclpy.executors import MultiThreadedExecutor
from control.launch_legs_motion import LegMotionNode

def main(args=None):
    rclpy.init(args=args)

    node_left = TrajectoryExecutorNode(TrajectoryExecutorSide.LEFT)
    node_right = TrajectoryExecutorNode(TrajectoryExecutorSide.RIGHT)
    leg_motion_node = LegMotionNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node_left)
    executor.add_node(node_right)
    executor.add_node(leg_motion_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node_left.destroy_node()
        node_right.destroy_node()
        rclpy.shutdown()
