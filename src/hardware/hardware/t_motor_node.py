import rclpy
import rclpy.node
from std_msgs.msg import Float32, Float32MultiArray
import axel_planner
from mini_cheetah_tmotor_can.src.motor_driver.canmotorlib import CanMotorController


DICT_MOTOR_ID = {
    "MOT_HIP_R": 0x2,
    "MOT_KNEE_R": 0x3,
}

CAN_SOCKET = "can0"


class TMotorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("t_motor_node")
        self.get_logger().info("T Motor Node has been started.")
        self.__is_motor_enabled = False
        self.hip_mot_PD_gain = {
            "P": 10.0,
            "D": 1,
        }  # Proportional and Derivative gains for hip motor
        self.knee_mot_PD_gain = {
            "P": 10.0,
            "D": 1,
        }  # Proportional and Derivative gains for knee motor

        self.dict_des_pos = {
            "MOT_HIP_R": 0.0,
            "MOT_KNEE_R": 0.0,
        }

        self.dict_act_pos_vel_current = {
            "MOT_HIP_R": (0.0, 0.0, 0.0),
            "MOT_KNEE_R": (0.0, 0.0, 0.0),
        }

        self.subscriber_des_pos = self.create_subscription(
            Float32MultiArray, "t_motor/desired_position", self.des_pos_callback, 10
        )

        self.pub_act_pos = self.create_publisher(
            Float32MultiArray, "t_motor/actual_position", 10
        )
        self.pub_act_vel = self.create_publisher(
            Float32MultiArray, "t_motor/actual_velocity", 10
        )
        self.pub_act_current = self.create_publisher(
            Float32MultiArray, "t_motor/actual_current", 10
        )

        self.timer_communicate_motor = self.create_timer(
            0.1, self.send_t_motor_callbacks
        )

        self.timer_publish_telemetry = self.create_timer(
            0.1, self.publish_actual_telemetry
        )

        self.motor_hip_r = CanMotorController(
            CAN_SOCKET, DICT_MOTOR_ID["MOT_HIP_R"], "AK10_9_V1p1"
        )
        self.motor_knee_r = CanMotorController(
            CAN_SOCKET, DICT_MOTOR_ID["MOT_KNEE_R"], "AK10_9_V1p1"
        )

    def des_pos_callback(self, msg: Float32MultiArray):
        des_pos_list = msg.data
        self.get_logger().debug(f"Received desired positions: {des_pos_list}")
        self.dict_des_pos["MOT_HIP_R"] = des_pos_list[0]
        self.dict_des_pos["MOT_KNEE_R"] = des_pos_list[1]

        if not self.__is_motor_enabled:
            self.motor_hip_r.enable_motor()
            self.motor_knee_r.enable_motor()
            self.__is_motor_enabled = True

    def send_t_motor_callbacks(self):
        pos_hip_r, vel_hip_r, curr_hip_r = self.motor_hip_r.send_rad_command(
            self.dict_des_pos["MOT_HIP_R"],
            0.0,
            self.hip_mot_PD_gain["P"],
            self.hip_mot_PD_gain["D"],
            0.0,
        )
        pos_knee_r, vel_knee_r, curr_knee_r = self.motor_knee_r.send_rad_command(
            self.dict_des_pos["MOT_KNEE_R"],
            0.0,
            self.knee_mot_PD_gain["P"],
            self.knee_mot_PD_gain["D"],
            0.0,
        )

        self.dict_act_pos_vel_current["MOT_HIP_R"] = (pos_hip_r, vel_hip_r, curr_hip_r)
        self.dict_act_pos_vel_current["MOT_KNEE_R"] = (
            pos_knee_r,
            vel_knee_r,
            curr_knee_r,
        )

    def publish_actual_telemetry(self):
        msg_pos = Float32MultiArray()
        msg_pos.data = [
            self.dict_act_pos_vel_current["MOT_HIP_R"][0],
            self.dict_act_pos_vel_current["MOT_KNEE_R"][0],
        ]

        self.pub_act_pos.publish(msg_pos)
        self.get_logger().debug(f"Published actual positions: {msg_pos.data}")

        msg_curr = Float32MultiArray()
        msg_curr.data = [
            self.dict_act_pos_vel_current["MOT_HIP_R"][2],
            self.dict_act_pos_vel_current["MOT_KNEE_R"][2],
        ]

        self.pub_act_current.publish(msg_curr)
        self.get_logger().debug(f"Published actual currents: {msg_curr.data}")


def main(args=None):

    rclpy.init(args=args)

    node = TMotorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
