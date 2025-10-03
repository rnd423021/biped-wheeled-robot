import pinocchio as pin
import numpy as np
from pathlib import Path

class BipedWheeledLeg():
    def __init__(
        self,
        mjcf_path=Path("biped_wheeled_leg/biped_wheeled_leg.xml")
    ):

        self.model = pin.buildModelFromMJCF(mjcf_path)
        init_pose = np.zeros(self.model.nq)
        self.data = self.model.createData()
        pin.forwardKinematics(self.model, self.data, init_pose)
        pin.updateFramePlacements(self.model, self.data)

        frames = ["hip_pitch_link", "knee_pitch_link", "ankle_wheel_link"]
        pos = []
        for frame in frames:
            frame_id = self.model.getFrameId(frame)
            pos.append(self.data.oMf[frame_id].translation)

        self.L1 = np.linalg.norm(pos[1] - pos[0])
        self.L2 = np.linalg.norm(pos[2] - pos[1])
        self.x_zero, self.y_zero = pos[2][0], pos[2][1]
        self.hip_pitch_zero, self.knee_pitch_zero = 0, 0
        self.hip_pitch_zero, self.knee_pitch_zero = self.ik_solve(self.x_zero, self.y_zero)
        self.hip_motor, self.knee_motor = 0, 0
        self.hip_pitch, self.knee_pitch = 0, 0

    def ik_solve(self, x, y):
        r2 = x*x + y*y
        r  = np.sqrt(r2)
        if r > self.L1 + self.L2 + 1e-9 or r < abs(self.L1 - self.L2) - 1e-9:
            raise ValueError("The target is geometrically unreachable.")

        c2 = (r2 - self.L1*self.L1 - self.L2*self.L2) / (2*self.L1*self.L2)
        c2 = np.clip(c2, -1.0, 1.0)
        s2 = np.sqrt(max(0.0, 1 - c2*c2))
        s2 = -s2

        self.knee_pitch = np.arctan2(s2, c2) - self.knee_pitch_zero
        self.hip_pitch = np.arctan2(y, x) - np.arctan2(self.L2*s2, self.L1 + self.L2*c2)  - self.hip_pitch_zero

        self.hip_motor = self.hip_pitch
        self.knee_motor = self.hip_pitch + self.knee_pitch
        return self.hip_motor, self.knee_motor


