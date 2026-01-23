import time
import mujoco
import mujoco.viewer
import numpy as np
from axel_planner.biped_wheeled_leg import BipedWheeledLeg


class OneLegRenderer:
    def __init__(self, model_path):
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        mj_data = mujoco.MjData(mj_model)
        self.mj_model = mj_model
        self.mj_data = mj_data

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.ref_qpos_knee_crank = self.mj_model.joint("knee_crank_joint").qpos0[0]

    def init_render(self):
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.y_zero_marker = self.mj_data.site("flag").xpos[1].copy()
    
    
    def render(self, hip_motor, knee_motor, x_target, z_target):

        self.mj_data.joint("knee_crank_joint").qpos = knee_motor
        self.mj_data.joint("hip_pitch_joint").qpos = hip_motor

        # Closed kinematics:
        self.mj_data.joint("knee_u_rod_joint").qpos = -(
            knee_motor + self.ref_qpos_knee_crank
        )
        self.mj_data.joint("knee_b_rod_joint").qpos = -knee_motor + hip_motor
        self.mj_data.joint("knee_ternery_joint").qpos = -(-knee_motor + hip_motor)

        # Open kinematics:
        self.mj_data.joint("knee_pitch_joint").qpos = -(-knee_motor + hip_motor)
        self.mj_data.joint("flag").qpos = [
            x_target,
            self.y_zero_marker,
            z_target,
            1,
            0,
            0,
            0,
        ]
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        self.viewer.sync()

    def forward_mujoco(self, hip_motor, knee_motor):

        self.mj_data.joint("knee_crank_joint").qpos = knee_motor
        self.mj_data.joint("hip_pitch_joint").qpos = hip_motor

        # Closed kinematics:
        self.mj_data.joint("knee_u_rod_joint").qpos = -(
            knee_motor + self.ref_qpos_knee_crank
        )
        self.mj_data.joint("knee_b_rod_joint").qpos = -knee_motor + hip_motor
        self.mj_data.joint("knee_ternery_joint").qpos = -(-knee_motor + hip_motor)

        # Open kinematics:
        self.mj_data.joint("knee_pitch_joint").qpos = -(-knee_motor + hip_motor)
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        ee_id = self.mj_data.body("ankle_wheel_link").id
        x_pos = self.mj_data.xpos[ee_id][0]
        z_pos = self.mj_data.xpos[ee_id][2]
        return x_pos, z_pos




if __name__ == "__main__":
    leg_model = BipedWheeledLeg()
    renderer = OneLegRenderer("biped_wheeled_leg/biped_wheeled_leg.xml")
    renderer.init_render()

    x_targets = np.linspace(leg_model.x_zero, -0.1, 20)
    z_targets = np.linspace(leg_model.y_zero, -0.6, 20)

    while True:
        for x_target, z_target in zip(x_targets, z_targets):
            hip_motor, knee_motor = leg_model.ik_solve(x_target, z_target)
            renderer.render(hip_motor, knee_motor, x_target, z_target)
            time.sleep(0.05)
        time.sleep(10.0)
