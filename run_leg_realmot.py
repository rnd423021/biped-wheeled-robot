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
from mini_cheetah_tmotor_can.src.motor_driver.canmotorlib import CanMotorController

CAN_MOTOR_ID_HIP = 0x2
CAN_MOTOR_ID_KNEE = 0x3


class TrajectoryExecutorStatus(enum.Enum):
    IDLE = 0
    RUNNING_TRAJ = 1
    PAUSED = 2


class TrajectoryExecutor:
    def __init__(self):

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


def traj_loop(
    executor: TrajectoryExecutor,
    hip_motor: CanMotorController,
    knee_motor: CanMotorController,
):
    """Thread function for trajectory execution callback"""
 
    while True:
        start_time_iteration = time.perf_counter()
        status = executor.status

        # Skip if not running
        if status != TrajectoryExecutorStatus.RUNNING_TRAJ:
            time.sleep(executor.traj_dt)
            continue

        # Initialize trajectory start time on first iteration
        if executor.start_time_traj is None:
            executor.start_time_traj = time.time()

        # Calculate current position in trajectory
        elapsed_time = time.time() - executor.start_time_traj
        traj_index = min(int(elapsed_time / executor.traj_dt), executor.num_points - 1)

        # Send commands to motors
        current_q = executor.q_values_traj[traj_index]
        current_xy = executor.xy_traj[traj_index]

        pos_hip, vel_hip, curr_hip = hip_motor.send_rad_command(current_q[0], 0.0, 100.0, 10, 0.0)
        pos_knee, vel_knee, curr_knee = knee_motor.send_rad_command(current_q[1], 0.0, 100.0, 10, 0.0)

        # Store current commanded position for logging in simulation loop
        executor.current_commanded_q = current_q

        # Stop when trajectory is complete
        if traj_index >= executor.num_points - 1:
            executor.start_time_traj = None
            executor.status = TrajectoryExecutorStatus.IDLE

        if executor.traj_dt - (time.perf_counter() - start_time_iteration) > 0:
            time.sleep(executor.traj_dt - (time.perf_counter() - start_time_iteration))
        else:
            print("Warning: Trajectory loop overran desired dt")


if __name__ == "__main__":

    XY_START_SQUAT = np.array([-0.148, -0.23])
    XY_END_SQUAT = np.array([-0.148, -0.55])

    traj_executor = TrajectoryExecutor()
    leg_biped = BipedWheeledLeg()

    CAN_SOCKET = "can0"
    CAN_MOTOR_ID_HIP = 0x2
    CAN_MOTOR_ID_KNEE = 0x3

    motor_hip = CanMotorController(CAN_SOCKET, CAN_MOTOR_ID_HIP, "AK10_9_V1p1")
    motor_knee = CanMotorController(CAN_SOCKET, CAN_MOTOR_ID_KNEE, "AK10_9_V1p1")
    motor_hip.enable_motor()
    motor_knee.enable_motor()


    trajectory_execution_thread = threading.Thread(
        target=traj_loop, args=(traj_executor, motor_hip, motor_knee), daemon=True
    )
    

    act_position_hip, _, _ = motor_hip.send_rad_command(0, 0.0, 0.0, 0.0, 0.0)
    act_position_knee, _, _ = motor_knee.send_rad_command(0, 0.0, 0.0, 0.0, 0.0)

    initial_x, initial_y = leg_biped.forward_kinematics(
        act_position_knee, act_position_hip
    )

    x_traj, y_traj, time_traj, num_points, traj_dt = get_trajectory(
        [initial_x, initial_y],
        XY_START_SQUAT,
        traj_velocity=0.3,
        traj_point_per_meter=600,
    )

    x_traj_2, y_traj_2, time_traj_2, num_points_2, traj_dt_2 = get_trajectory(
        XY_START_SQUAT,
        XY_END_SQUAT,
        traj_velocity=0.3,
        traj_point_per_meter=600,
    )

    traj_executor.set_trajectory(
        xy_traj=np.array([x_traj, y_traj]).T,
        time_traj=time_traj,
        num_points=num_points,
        traj_dt=traj_dt,
    )
 
    traj_executor.start()
    trajectory_execution_thread.start()

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

    # # Get and analyze tracking errors
    # error_data = traj_executor.get_error_data()
    # save_error_data(error_data, "trajectory_errors.npz")
    # print_error_statistics(error_data)
    # plot_tracking_errors(error_data, show=True, save_path="trajectory_errors.png")
