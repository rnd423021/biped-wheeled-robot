import sys
import time
import numpy as np
 
from src_non_ros.mini_cheetah_tmotor_can.src.motor_driver.canmotorlib import CanMotorController

 
 
CAN_SOCKET = "can0"
CAN_MOTOR_ID_HIP = 0x2
CAN_MOTOR_ID_KNEE = 0x3

motor_hip = CanMotorController(CAN_SOCKET, CAN_MOTOR_ID_HIP, "AK10_9_V1p1")
motor_knee = CanMotorController(CAN_SOCKET, CAN_MOTOR_ID_KNEE, "AK10_9_V1p1") 

motor_hip.enable_motor()
motor_knee.enable_motor()


pos, vel, curr = motor_knee.send_deg_command(0, 0.0, 20.0, 1, 0.0)
pos, vel, curr = motor_hip.send_deg_command(0, 0.0, 20.0, 1, 0.0)
time.sleep(2)
 


motor_knee.disable_motor()
motor_hip.disable_motor()