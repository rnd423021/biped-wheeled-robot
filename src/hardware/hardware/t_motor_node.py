import rclpy
import rclpy.node
from std_msgs.msg import Float32, Float32MultiArray
import axel_planner
from mini_cheetah_tmotor_can.src.motor_driver.canmotorlib import CanMotorController
from communication.msg import MotorCommand
from std_srvs.srv import SetBool

DICT_MOTOR_ID = {
    "MOT_HIP_R": 0x2,
    "MOT_KNEE_R": 0x3,
    "MOT_HIP_L": 0x4,
    "MOT_KNEE_L": 0x1,
}

CAN_SOCKET = "can0"


class TMotorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("t_motor_node")
        self.get_logger().info("T Motor Node has been started.")
        self.is_left_side_active = True

        self.__is_motor_enabled_right = False
        self.__is_motor_enabled_left = False

        self.hip_mot_PD_gain = {
            "P": 50.0,
            "D": 1.0,
        }  # Proportional and Derivative gains for hip motor
        self.knee_mot_PD_gain = {
            "P": 50.0,
            "D": 1.0,
        }  # Proportional and Derivative gains for knee motor
        self.__saved_knee_mot_P_gain = self.knee_mot_PD_gain["P"]
        self.__saved_knee_mot_D_gain = self.knee_mot_PD_gain["D"]

        self.dict_des_pos = {
            "MOT_HIP_R": 0.0,
            "MOT_KNEE_R": 0.0,
            "MOT_HIP_L": 0.0,
            "MOT_KNEE_L": 0.0,
        }

        self.dict_act_pos_vel_current = {
            "MOT_HIP_R": (0.0, 0.0, 0.0),
            "MOT_KNEE_R": (0.0, 0.0, 0.0),
            "MOT_HIP_L": (0.0, 0.0, 0.0),
            "MOT_KNEE_L": (0.0, 0.0, 0.0),
        }

        self.subscriber_des_pos_right = self.create_subscription(
            MotorCommand,
            "t_motor_right/desired_position",
            self.des_pos_callback_right,
            10,
        )

        self.subscriber_des_pos_left = self.create_subscription(
            MotorCommand,
            "t_motor_left/desired_position",
            self.des_pos_callback_left,
            10,
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
            0.02, self.publish_actual_telemetry
        )

        self.knee_motor_switch_srv = self.create_service(
            SetBool,
            "t_motor/enable_knee_motors",
            self.switch_knee_motors_torque_callback,
        )

        self.motor_hip_r = CanMotorController(
            CAN_SOCKET, DICT_MOTOR_ID["MOT_HIP_R"], "AK10_9_V1p1"
        )
        self.motor_knee_r = CanMotorController(
            CAN_SOCKET, DICT_MOTOR_ID["MOT_KNEE_R"], "AK10_9_V1p1"
        )
        if self.is_left_side_active:
            self.motor_hip_l = CanMotorController(
                CAN_SOCKET, DICT_MOTOR_ID["MOT_HIP_L"], "AK10_9_V1p1"
            )
            self.motor_knee_l = CanMotorController(
                CAN_SOCKET, DICT_MOTOR_ID["MOT_KNEE_L"], "AK10_9_V1p1"
            )

    def remove_initial_jerk_and_enable(self):

        if self.is_left_side_active:
            self.motor_hip_l.enable_motor()
            self.motor_knee_l.enable_motor()

    def des_pos_callback_right(self, msg: MotorCommand):

        if not self.__is_motor_enabled_right:
            self.motor_hip_r.enable_motor()
            self.motor_knee_r.enable_motor()
            self.__is_motor_enabled_right = True

        self.get_logger().debug(
            f"Received desired positions: hip {msg.hip_motor}, knee {msg.knee_motor}"
        )
        self.dict_des_pos["MOT_HIP_R"] = msg.hip_motor
        self.dict_des_pos["MOT_KNEE_R"] = msg.knee_motor

    def des_pos_callback_left(self, msg: MotorCommand):
        if not self.__is_motor_enabled_left:
            self.motor_hip_l.enable_motor()
            self.motor_knee_l.enable_motor()
            self.__is_motor_enabled_left = True

        self.get_logger().debug(
            f"Received desired positions: hip {msg.hip_motor}, knee {msg.knee_motor}"
        )
        self.dict_des_pos["MOT_HIP_L"] = msg.hip_motor
        self.dict_des_pos["MOT_KNEE_L"] = msg.knee_motor

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
        if self.is_left_side_active:
            pos_hip_l, vel_hip_l, curr_hip_l = self.motor_hip_l.send_rad_command(
                -self.dict_des_pos["MOT_HIP_L"],
                0.0,
                self.hip_mot_PD_gain["P"],
                self.hip_mot_PD_gain["D"],
                0.0,
            )
            pos_knee_l, vel_knee_l, curr_knee_l = self.motor_knee_l.send_rad_command(
                -self.dict_des_pos["MOT_KNEE_L"],
                0.0,
                self.knee_mot_PD_gain["P"],
                self.knee_mot_PD_gain["D"],
                0.0,
            )

            self.dict_act_pos_vel_current["MOT_HIP_L"] = (
                -pos_hip_l,
                -vel_hip_l,
                curr_hip_l,
            )
            self.dict_act_pos_vel_current["MOT_KNEE_L"] = (
                -pos_knee_l,
                -vel_knee_l,
                curr_knee_l,
            )

    def publish_actual_telemetry(self):
        msg_pos = Float32MultiArray()
        msg_pos.data = [
            self.dict_act_pos_vel_current["MOT_HIP_R"][0],
            self.dict_act_pos_vel_current["MOT_KNEE_R"][0],
        ]
        if self.is_left_side_active:
            msg_pos.data.extend(
                [
                    self.dict_act_pos_vel_current["MOT_HIP_L"][0],
                    self.dict_act_pos_vel_current["MOT_KNEE_L"][0],
                ]
            )

        self.pub_act_pos.publish(msg_pos)
        self.get_logger().debug(f"Published actual positions: {msg_pos.data}")

        msg_curr = Float32MultiArray()
        msg_curr.data = [
            self.dict_act_pos_vel_current["MOT_HIP_R"][2],
            self.dict_act_pos_vel_current["MOT_KNEE_R"][2],
        ]
        if self.is_left_side_active:
            msg_curr.data.extend(
                [
                    self.dict_act_pos_vel_current["MOT_HIP_L"][2],
                    self.dict_act_pos_vel_current["MOT_KNEE_L"][2],
                ]
            )

        self.pub_act_current.publish(msg_curr)
        self.get_logger().debug(f"Published actual currents: {msg_curr.data}")

    def turn_off_commands(self):
        self.motor_hip_r.send_rad_command(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.motor_knee_r.send_rad_command(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.motor_hip_r.disable_motor()
        self.motor_knee_r.disable_motor()
        if self.is_left_side_active:
            self.motor_hip_l.send_rad_command(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            self.motor_knee_l.send_rad_command(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            self.motor_hip_l.disable_motor()
            self.motor_knee_l.disable_motor()

    def switch_knee_motors_torque_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ):
        if not request.data:
            if self.knee_mot_PD_gain["P"] != 0.0:
                self.__saved_knee_mot_P_gain = self.knee_mot_PD_gain["P"]
                self.knee_mot_PD_gain["P"] = 0.0
            if self.knee_mot_PD_gain["D"] != 0.0:
                self.__saved_knee_mot_D_gain = self.knee_mot_PD_gain["D"]
                self.knee_mot_PD_gain["D"] = 0.0
        else:
            self.knee_mot_PD_gain["P"] = self.__saved_knee_mot_P_gain
            self.knee_mot_PD_gain["D"] = self.__saved_knee_mot_D_gain
            self.dict_des_pos["MOT_KNEE_R"] = self.dict_act_pos_vel_current["MOT_KNEE_R"][0]
            self.dict_des_pos["MOT_KNEE_L"] = self.dict_act_pos_vel_current["MOT_KNEE_L"][0] 

        response.success = True
        return response


def main(args=None):

    rclpy.init(args=args)

    node = TMotorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.turn_off_commands()
    finally:
        node.turn_off_commands()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
