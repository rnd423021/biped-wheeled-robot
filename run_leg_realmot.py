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
import threading
import enum


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


class TrajectoryExecutorStatus(enum.Enum):
    IDLE = 0
    RUNNING_TRAJ = 1
    PAUSED = 2


class TrajectoryExecutor:
    def __init__(self, mock_motors: MotorMock):
        self.mock_motors = mock_motors
        self.leg_model = BipedWheeledLeg()
        self.status = TrajectoryExecutorStatus.IDLE
        self.lock = threading.Lock()

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
        with self.lock:
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
        with self.lock:
            if self.q_values_traj is not None:
                self.start_time_traj = None
                self.status = TrajectoryExecutorStatus.RUNNING_TRAJ

    def pause(self):
        """Pause trajectory execution"""
        with self.lock:
            self.status = TrajectoryExecutorStatus.PAUSED

    def stop(self):
        """Stop trajectory execution"""
        with self.lock:
            self.start_time_traj = None
            self.status = TrajectoryExecutorStatus.IDLE

    def get_error_data(self):
        """Get error tracking data"""
        with self.lock:
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


def plot_tracking_errors(error_data: dict, show: bool = True, save_path: str = None):
    """Plot trajectory tracking errors

    Args:
        error_data: Dictionary with 'error', 'time', 'desired_q', 'actual_q' arrays
        show: Whether to display the plot
        save_path: Path to save the plot (e.g., 'errors.png'). If None, don't save
    """
    if len(error_data["error"]) == 0:
        print("No error data to plot")
        return

    error_array = error_data["error"]
    time_array = error_data["time"]
    desired_q = error_data["desired_q"]
    actual_q = error_data["actual_q"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Plot joint errors
    axes[0].plot(time_array, error_array[:, 0], label="Hip Error", color="red")
    axes[0].plot(time_array, error_array[:, 1], label="Knee Error", color="blue")
    axes[0].set_ylabel("Error (rad)")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("Joint Position Tracking Error")

    # Plot hip joint tracking
    axes[1].plot(
        time_array, desired_q[:, 0], label="Desired Hip", linestyle="--", color="red"
    )
    axes[1].plot(time_array, actual_q[:, 0], label="Actual Hip", color="red")
    axes[1].set_ylabel("Hip Position (rad)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title("Hip Joint Tracking")

    # Plot knee joint tracking
    axes[2].plot(
        time_array, desired_q[:, 1], label="Desired Knee", linestyle="--", color="blue"
    )
    axes[2].plot(time_array, actual_q[:, 1], label="Actual Knee", color="blue")
    axes[2].set_ylabel("Knee Position (rad)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_title("Knee Joint Tracking")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_error_data(error_data: dict, filename: str = "tracking_errors.npz"):
    """Save error tracking data to file

    Args:
        error_data: Dictionary with 'error', 'time', 'desired_q', 'actual_q' arrays
        filename: Output filename (numpy .npz format)
    """
    if len(error_data["error"]) == 0:
        print("No error data to save")
        return

    np.savez(filename, **error_data)
    print(f"Error data saved to {filename}")


def print_error_statistics(error_data: dict):
    """Print tracking error statistics

    Args:
        error_data: Dictionary with 'error', 'time', 'desired_q', 'actual_q' arrays
    """
    if len(error_data["error"]) == 0:
        print("No error data available")
        return

    error_array = error_data["error"]

    print(f"\nTracking Error Statistics:")
    print(
        f"Hip - Mean Error: {np.mean(np.abs(error_array[:, 0])):.4f} rad, Max Error: {np.max(np.abs(error_array[:, 0])):.4f} rad"
    )
    print(
        f"Knee - Mean Error: {np.mean(np.abs(error_array[:, 1])):.4f} rad, Max Error: {np.max(np.abs(error_array[:, 1])):.4f} rad"
    )


def simulation_loop(executor: TrajectoryExecutor, mock_motors: MotorMock):
    """Thread function for MuJoCo simulation stepping and rendering"""
    with mujoco.viewer.launch_passive(
        mock_motors.mj_model, mock_motors.mj_data
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()

            with executor.lock:
                mujoco.mj_step(mock_motors.mj_model, mock_motors.mj_data)

                # Log tracking error if trajectory is running
                if (
                    executor.status == TrajectoryExecutorStatus.RUNNING_TRAJ
                    and executor.current_commanded_q is not None
                ):
                    act_hip = mock_motors.mj_data.joint("hip_pitch_joint").qpos[0]
                    act_knee = mock_motors.mj_data.joint("knee_crank_joint").qpos[0]

                    current_q = executor.current_commanded_q

                    elapsed_time = mock_motors.mj_data.time - executor.start_time_traj

                    error = np.array([act_hip - current_q[0], act_knee - current_q[1]])
                    executor.error_log.append(error)
                    executor.time_log.append(mock_motors.mj_data.time)
                    executor.desired_q_log.append(current_q)
                    executor.actual_q_log.append(np.array([act_hip, act_knee]))

            viewer.sync()

            # Maintain real-time simulation speed
            time_until_next_step = mock_motors.mj_model.opt.timestep - (
                time.time() - step_start
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def traj_loop(executor: TrajectoryExecutor, motor_mock: MotorMock):
    """Thread function for trajectory execution callback"""
    while True:
        with executor.lock:
            status = executor.status

            # Skip if not running
            if status != TrajectoryExecutorStatus.RUNNING_TRAJ:
                time.sleep(executor.traj_dt)
                continue

            # Initialize trajectory start time on first iteration
            if executor.start_time_traj is None:
                executor.start_time_traj = motor_mock.mj_data.time

            # Calculate current position in trajectory
            elapsed_time = motor_mock.mj_data.time - executor.start_time_traj
            traj_index = min(
                int(elapsed_time / executor.traj_dt), executor.num_points - 1
            )

            # Send commands to motors
            current_q = executor.q_values_traj[traj_index]
            current_xy = executor.xy_traj[traj_index]
            mock_motors.mj_data.joint("flag").qpos = [
                current_xy[0],
                -0.119,
                current_xy[1],
                1,
                0,
                0,
                0,
            ]
            executor.mock_motors.send_rad_command(current_q[0], mot_id=CAN_MOTOR_ID_HIP)
            executor.mock_motors.send_rad_command(
                current_q[1], mot_id=CAN_MOTOR_ID_KNEE
            )

            # Store current commanded position for logging in simulation loop
            executor.current_commanded_q = current_q

            # Stop when trajectory is complete
            if traj_index >= executor.num_points - 1:
                executor.start_time_traj = None
                executor.status = TrajectoryExecutorStatus.IDLE

        time.sleep(executor.traj_dt)


if __name__ == "__main__":

    XY_START_SQUAT = np.array([-0.148, -0.286])
    XY_END_SQUAT = np.array([-0.148, -0.6])

    mock_motors = MotorMock()
    traj_executor = TrajectoryExecutor(mock_motors)
    leg_biped = BipedWheeledLeg()

    sim_thread = threading.Thread(
        target=simulation_loop, args=(traj_executor, mock_motors), daemon=True
    )

    # Thread for trajectory callback
    trajectory_execution_thread = threading.Thread(
        target=traj_loop, args=(traj_executor, mock_motors), daemon=True
    )

    mujoco.mj_forward(
        traj_executor.mock_motors.mj_model, traj_executor.mock_motors.mj_data
    )

    act_position_hip, _, _ = traj_executor.mock_motors.send_rad_command(
        0, CAN_MOTOR_ID_HIP
    )
    act_position_knee, _, _ = traj_executor.mock_motors.send_rad_command(
        0, CAN_MOTOR_ID_KNEE
    )

    initial_x, initial_y = leg_biped.forward_kinematics(
        act_position_knee, act_position_hip
    )

    x_traj, y_traj, time_traj, num_points, traj_dt = get_trajectory(
        [initial_x, initial_y],
        XY_START_SQUAT,
        traj_velocity=0.8,
        traj_point_per_meter=100,
    )

    x_traj_2, y_traj_2, time_traj_2, num_points_2, traj_dt_2 = get_trajectory(
        XY_START_SQUAT,
        XY_END_SQUAT,
        traj_velocity=0.8,
        traj_point_per_meter=100,
    )

    traj_executor.set_trajectory(
        xy_traj=np.array([x_traj, y_traj]).T,
        time_traj=time_traj,
        num_points=num_points,
        traj_dt=traj_dt,
    )

    # Start simulation and callback threads
    trajectory_execution_thread.start()
    sim_thread.start()

    #
    traj_executor.start()

    # Wait until trajectory completes
    while traj_executor.status == TrajectoryExecutorStatus.RUNNING_TRAJ:
        time.sleep(0.1)

    traj_executor.set_trajectory(
        xy_traj=np.array([x_traj_2, y_traj_2]).T,
        time_traj=time_traj_2,
        num_points=num_points_2,
        traj_dt=traj_dt_2,
    )
    input("Press Enter to start second trajectory...")
    traj_executor.start()
    while traj_executor.status == TrajectoryExecutorStatus.RUNNING_TRAJ:
        time.sleep(0.1)

    # Get and analyze tracking errors
    error_data = traj_executor.get_error_data()
    save_error_data(error_data, "trajectory_errors.npz")
    print_error_statistics(error_data)
    plot_tracking_errors(error_data, show=True, save_path="trajectory_errors.png")
