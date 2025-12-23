import time
import mujoco
import mujoco.viewer
import numpy as np
from axel_planner.biped_wheeled_leg import BipedWheeledLeg


model = mujoco.MjModel.from_xml_path("biped_wheeled_leg/biped_wheeled_leg.xml")
data = mujoco.MjData(model)
ee_id = data.body("ankle_wheel_link").id


 
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
 
        mujoco.mj_step(model, data)
        viewer.sync()
        if data.time > 2:
            data.xfrc_applied[ee_id][:3] = [0, 0, 100] 


