from control.one_leg_planner import (
    TrajectoryExecutorNode,
    TrajectoryExecutorSide,
)
import rclpy
from rclpy.executors import MultiThreadedExecutor


def main(args=None):
    rclpy.init(args=args)

    node_left = TrajectoryExecutorNode(TrajectoryExecutorSide.LEFT)
    node_right = TrajectoryExecutorNode(TrajectoryExecutorSide.RIGHT)

    executor = MultiThreadedExecutor()
    executor.add_node(node_left)
    executor.add_node(node_right)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node_left.destroy_node()
        node_right.destroy_node()
        rclpy.shutdown()
