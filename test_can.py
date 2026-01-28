import sys
import time
import numpy as np
 
from src_non_ros.mini_cheetah_tmotor_can.src.motor_driver.canmotorlib import CanMotorController

 
 
CAN_SOCKET = "can0"
CAN_MOTOR_ID_HIP = 0x4
CAN_MOTOR_ID_KNEE = 0x1

motor_hip = CanMotorController(CAN_SOCKET, CAN_MOTOR_ID_HIP, "AK10_9_V1p1")
motor_knee = CanMotorController(CAN_SOCKET, CAN_MOTOR_ID_KNEE, "AK10_9_V1p1") 

motor_knee.enable_motor()
motor_hip.enable_motor()

time.sleep(1)

  
pos, vel, curr = motor_knee.send_deg_command(-10, 0.0, 50.0, 1, 0.0)
pos, vel, curr = motor_hip.send_deg_command(-20, 0.0, 50.0, 1, 0.0)
time.sleep(1)
 

 