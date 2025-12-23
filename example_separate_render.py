import time
import numpy as np
from axel_planner.biped_wheeled_leg import BipedWheeledLeg
from axel_planner.one_leg_reneder import OneLegRenderer
import mujoco

 


if __name__ == "__main__":
    leg_model = BipedWheeledLeg()
    renderer = OneLegRenderer("biped_wheeled_leg/biped_wheeled_leg.xml")

    mj_model = mujoco.MjModel.from_xml_path("biped_wheeled_leg/biped_wheeled_leg.xml")
    mj_data = mujoco.MjData(mj_model)
    

    renderer.init_render()

    x_targets = np.linspace(leg_model.x_zero, -0.05, 20)
    z_targets = np.linspace(leg_model.y_zero, -0.5, 20)

    while True:
        for x_target, z_target in zip(x_targets, z_targets):
            hip_motor, knee_motor = leg_model.ik_solve(x_target, z_target)
            jacoba = leg_model._jacobian(leg_model.hip_pitch, leg_model.knee_pitch)
            renderer.render(hip_motor, knee_motor, x_target, z_target)
            time.sleep(0.05)
        time.sleep(4.0)
