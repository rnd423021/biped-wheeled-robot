import time
import mujoco
import mujoco.viewer
import numpy as np
from biped_wheeled_leg import BipedWheeledLeg

model = mujoco.MjModel.from_xml_path('biped_wheeled_leg/biped_wheeled_leg.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

biped = BipedWheeledLeg()

x_zero, z_zero = biped.x_zero, biped.y_zero
y_zero = data.site('flag').xpos[1].copy()

hip_motor, knee_motor = biped.ik_solve(x_zero, z_zero)

std = 0.1
dt = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        for i in range(1000):
            t0 = time.perf_counter()
            x, z = x_zero + np.random.uniform(-std, std), z_zero + np.random.uniform(-std, std)
            hip_motor, knee_motor = biped.ik_solve(x, z)
            dt.append(time.perf_counter() - t0)

            # Motors:
            data.joint("knee_crank_joint").qpos = knee_motor
            data.joint("hip_pitch_joint").qpos = hip_motor
            
            # Closed kinematics:
            data.joint("knee_u_rod_joint").qpos = -knee_motor
            data.joint("knee_b_rod_joint").qpos = biped.knee_pitch
            data.joint("knee_ternery_joint").qpos = -biped.knee_pitch
            
            # Open kinematics:
            data.joint("knee_pitch_joint").qpos = -biped.knee_pitch

            # data.joint("knee_pitch_joint").qpos = - biped.knee_pitch
            data.joint('flag').qpos = [x, y_zero, z, 1, 0, 0, 0]

            mujoco.mj_step(model, data)
            time.sleep(0.05)
            viewer.sync()
        break

print("Mean dt to calculate IK:", np.mean(dt), "sec")
print("Mean frequency to calculate IK:", 1/np.mean(dt), "Hz")
