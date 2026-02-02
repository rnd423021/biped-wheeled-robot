import enum
import numpy as np
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String
from geometry_msgs.msg import Point

from communication.srv import SetTargetPosition, SetRollCommand, SetHeight

from axel_planner.biped_wheeled_leg import BipedWheeledLeg
from axel_planner.planner import get_trajectory
from control.one_leg_planner import TrajectoryExecutorSide


class LegMotionNode(Node):
    def __init__(self):
        super().__init__("leg_motion_node")

        self.set_end_effector_client_left = self.create_client(
            SetTargetPosition,
            f"leg_planner_{TrajectoryExecutorSide.LEFT.value}/set_target_position",
        )

        self.set_end_effector_client_right = self.create_client(
            SetTargetPosition,
            f"leg_planner_{TrajectoryExecutorSide.RIGHT.value}/set_target_position",
        )

        self.set_roll_srv = self.create_service(
            SetRollCommand,
            "legs_motion/set_roll_command",
            self.set_roll_command_callback,
        )

        self.set_height_srv = self.create_service(
            SetHeight,
            "legs_motion/set_height",
            self.set_height_callback,
        )

        self.call_in_progress = False
        self.pending_futures_roll = []

        self.mid_point = [-0.1, -0.3]  # meters
        self.body_width = 0.6  # meters

    def set_height_callback(
        self, request: SetHeight.Request, response: SetHeight.Response
    ):
        height = request.height
        mid_point = self.mid_point  # meters
        des_ee_pos_left = np.array([mid_point[0], -height])
        des_ee_pos_right = np.array([mid_point[0], -height])

        right_request = SetTargetPosition.Request()
        right_request.x = float(des_ee_pos_right[0])
        right_request.y = float(des_ee_pos_right[1])
        right_future = self.set_end_effector_client_right.call_async(right_request)
        right_future.add_done_callback(self.handle_response)

        left_request = SetTargetPosition.Request()
        left_request.x = float(des_ee_pos_left[0])
        left_request.y = float(des_ee_pos_left[1])
        left_future = self.set_end_effector_client_left.call_async(left_request)
        left_future.add_done_callback(self.handle_response)

        response.success = True
        return response

    def handle_response(self, future):
        response = future.result()
        self.get_logger().info(
            f"Response {type(response)} received: {response.success}"
        )

    def set_roll_command_callback(
        self, request: SetRollCommand.Request, response: SetRollCommand.Response
    ):
        roll_angle = request.roll
        desired_high_delta = np.tan(np.deg2rad(roll_angle)) * self.body_width
        mid_point = self.mid_point  # meters
        des_ee_pos_left = mid_point + np.array([0.0, desired_high_delta / 2])
        des_ee_pos_right = mid_point + np.array([0.0, -desired_high_delta / 2])

        right_request = SetTargetPosition.Request()
        right_request.x = float(des_ee_pos_right[0])
        right_request.y = float(des_ee_pos_right[1])
        right_future = self.set_end_effector_client_right.call_async(right_request)
        right_future.add_done_callback(self.handle_response)

        left_request = SetTargetPosition.Request()
        left_request.x = float(des_ee_pos_left[0])
        left_request.y = float(des_ee_pos_left[1])
        left_future = self.set_end_effector_client_left.call_async(left_request)
        left_future.add_done_callback(self.handle_response)

        response.success = True
        return response
