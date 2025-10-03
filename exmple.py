import time
import mujoco
import mujoco.viewer
from biped_wheeled_leg import BipedWheeledLeg

model = mujoco.MjModel.from_xml_path('biped_wheeled_leg/biped_wheeled_leg.xml')
data = mujoco.MjData(model)

biped = BipedWheeledLeg()
with mujoco.viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 60:
    step_start = time.time()
    mujoco.mj_step(model, data)
    time.sleep(0.001)

