from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt
from axel_planner.biped_wheeled_leg import BipedWheeledLeg
from axel_planner.one_leg_reneder import OneLegRenderer
import time
import mujoco
import mujoco.viewer
from copy import deepcopy
from axel_planner.planner import get_trajectory


CAN_MOTOR_ID_HIP = 0x2
CAN_MOTOR_ID_KNEE = 0x3


class MotorMock:
    def __init__(self, path_to_xml: str = "biped_wheeled_leg/biped_wheeled_leg.xml"):

        self.mj_model = mujoco.MjModel.from_xml_path(path_to_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.ref_qpos_knee_crank = self.mj_model.joint("knee_crank_joint").qpos0[0]

        start_knee_motor = 0.0
        start_hip_motor = 0.0

        self.mj_data.joint("knee_crank_joint").qpos = (
            start_knee_motor - self.ref_qpos_knee_crank
        )
        self.mj_data.joint("hip_pitch_joint").qpos = start_hip_motor

        # Closed kinematics:
        self.mj_data.joint("knee_u_rod_joint").qpos = -(
            start_knee_motor - self.ref_qpos_knee_crank
        )
        self.mj_data.joint("knee_b_rod_joint").qpos = (
            -start_knee_motor + start_hip_motor
        )
        self.mj_data.joint("knee_ternery_joint").qpos = -(
            -start_knee_motor + start_hip_motor
        )

        # Open kinematics:
        self.mj_data.joint("knee_pitch_joint").qpos = -(
            -start_knee_motor + start_hip_motor
        )

    def change_motor_pd_gains(self, Kp, Kd, mot_id):
        self.mj_model = deepcopy(self.mj_model)
        if mot_id == 0x2:
            self.mj_model.actuator_gainprm[0, 0] = Kp
            self.mj_model.actuator_biasprm[0, 2] = -Kp  # -kp (for position actuators)
            self.mj_model.actuator_biasprm[0, 1] = Kd
        elif mot_id == 0x3:
            self.mj_model.actuator_gainprm[1, 0] = Kp
            self.mj_model.actuator_gainprm[1, 2] = -Kp
            self.mj_model.actuator_biasprm[1, 1] = Kd
        new_mj_data = mujoco.MjData(self.mj_model)
        new_mj_data.qpos[:] = self.mj_data.qpos[:]
        new_mj_data.qvel[:] = self.mj_data.qvel[:]
        self.mj_data = new_mj_data

    def send_rad_command(self, position_in_rad, mot_id):

        can2ctrl = {CAN_MOTOR_ID_HIP: 0, CAN_MOTOR_ID_KNEE: 1}

        self.mj_data.ctrl[can2ctrl[mot_id]] = position_in_rad

        if mot_id == CAN_MOTOR_ID_HIP:
            act_position = self.mj_data.joint("knee_crank_joint").qpos
        elif mot_id == CAN_MOTOR_ID_KNEE:
            act_position = self.mj_data.joint("hip_pitch_joint").qpos

        return act_position, 0, 0


class TrajectoryExecutor:
    def __init__(self):
        self.is_execute_traj = False
        self.mock_motors = MotorMock()
        self.leg_model = BipedWheeledLeg()

    def set_trajectory(self, xy_traj, time_traj, num_points, traj_dt):
        self.xy_traj = xy_traj
        self.time_traj = time_traj
        self.num_points = num_points
        self.traj_dt = traj_dt
        self.is_running_traj = False
        self.q_values_traj = np.array(
            [self.leg_model.ik_solve(xy[0], xy[1]) for xy in self.xy_traj]
        )

    def enable_execute_traj(self):
        self.is_execute_traj = True

    def ruining_callback(self):

        if not self.is_running_traj:
            self.start_time_traj = time.perf_counter()

        if self.is_execute_traj:
            current_time = time.perf_counter() - self.start_time_traj
            traj_it = min(int(current_time / traj_dt), num_points - 1)
            current_q = self.q_values_traj[traj_it]
            mock_motors.send_rad_command(current_q[0], mot_id=CAN_MOTOR_ID_HIP)
            mock_motors.send_rad_command(current_q[0], mot_id=CAN_MOTOR_ID_HIP)


if __name__ == "__main__":

    XY_START_SQUAT = np.array([0.148, -0.386])
    XY_END_SQUAT = np.array([0, -0.6])

    mock_motors = MotorMock()
    leg_biped = BipedWheeledLeg()

    act_position_hip, _, _ = mock_motors.send_rad_command(0, CAN_MOTOR_ID_HIP)
    act_position_knee, _, _ = mock_motors.send_rad_command(0, CAN_MOTOR_ID_KNEE)

    initial_x, initial_y = leg_biped.forward_kinematics(
        act_position_hip, act_position_knee
    )

    x_traj, y_traj, time_traj, num_points, traj_dt = get_trajectory(
        XY_START_SQUAT, XY_END_SQUAT, traj_velocity=1.0, traj_point_per_meter=100
    )
