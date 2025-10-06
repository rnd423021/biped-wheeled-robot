import pinocchio as pin
import numpy as np
from pathlib import Path

class BipedWheeledLeg():
    """Kinematic model of a planar two-link biped leg with a wheel at the ankle.

    This class loads a Pinocchio model from a URDF file and provides
    analytical inverse kinematics (IK) and Jacobian computations for
    the hip and knee pitch joints. The leg is treated as a 2D mechanism
    in the sagittal plane (x–z).
    """
    def __init__(
        self,
        urdf_path=Path("biped_wheeled_leg/biped_wheeled_leg_open.urdf")
    ):
        """Initializes the BipedWheeledLeg model.

        Args:
            urdf_path (Path): Path to the URDF file describing the leg.
        """
        self.model = pin.buildModelFromUrdf(urdf_path)
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
        self.x_zero, self.y_zero = pos[2][0], pos[2][2]
        self.hip_pitch_zero, self.knee_pitch_zero = 0, 0
        self.ik_solve(self.x_zero, self.y_zero)
        self.hip_pitch_zero, self.knee_pitch_zero = self.hip_pitch, self.knee_pitch
        self.hip_motor, self.knee_motor = 0, 0
        self.hip_pitch, self.knee_pitch = 0, 0

    def ik_solve(self, x, y):
        """Computes inverse kinematics for a 2-link planar leg.

        Args:
            x (float): Target x-coordinate of the ankle in the leg plane.
            y (float): Target z-coordinate (vertical) of the ankle.

        Returns:
            tuple[float, float]: (hip_motor, knee_motor) joint commands.

        Raises:
            ValueError: If the target is geometrically unreachable.
        """
        r2 = x*x + y*y
        r  = np.sqrt(r2)
        if r > self.L1 + self.L2 + 1e-9 or r < abs(self.L1 - self.L2) - 1e-9:
            raise ValueError("The target is geometrically unreachable.")

        c2 = (r2 - self.L1*self.L1 - self.L2*self.L2) / (2*self.L1*self.L2)
        c2 = np.clip(c2, -1.0, 1.0)
        s2 = np.sqrt(max(0.0, 1 - c2*c2))
        s2 = -s2

        self.knee_pitch = np.arctan2(s2, c2) - self.knee_pitch_zero
        self.hip_pitch = np.arctan2(y, x) - np.arctan2(self.L2*s2, self.L1 + self.L2*c2) - self.hip_pitch_zero

        self.hip_motor = - self.hip_pitch
        self.knee_motor = - (self.knee_pitch + self.hip_pitch)
        return self.hip_motor, self.knee_motor

    def _jacobian(self, hip_pitch, knee_pitch):
        """Computes the analytical 2×2 Jacobian for the end-effector in the leg plane.

        Args:
            hip_pitch (float): Hip joint angle (rad).
            knee_pitch (float): Knee joint angle (rad).

        Returns:
            np.ndarray: 2×2 Jacobian matrix mapping joint velocities
                [hip_dot, knee_dot] to Cartesian velocity [x_dot, y_dot].
        """
        th1, th2 = hip_pitch + self.hip_pitch_zero, knee_pitch + self.knee_pitch_zero
        s1, c1 = np.sin(th1), np.cos(th1)
        s12, c12 = np.sin(th1+th2), np.cos(th1+th2)
        J = np.array([
            [-self.L1*s1 - self.L2*s12,    -self.L2*s12],
            [ self.L1*c1 + self.L2*c12,     self.L2*c12]
        ], dtype=float)
        return J

    def jacobian(self, hip_motor, knee_motor):
        """Computes the Jacobian with respect to motor coordinates.

        Converts the joint-space Jacobian to motor-space using
        the actuator-to-joint mapping defined in this model.

        Args:
            hip_motor (float): Hip motor angle (rad).
            knee_motor (float): Knee motor angle (rad).

        Returns:
            np.ndarray: 2×2 Jacobian mapping motor velocities
                [hip_motor_dot, knee_motor_dot] to Cartesian velocity [x_dot, y_dot].
        """
        th1 = -hip_motor
        th2 = hip_motor - knee_motor
        Jq  = self.jacobian_xy(th1, th2)
        A   = np.array([[-1.0, 0.0],
                        [ 1.0,-1.0]])
        return Jq @ A