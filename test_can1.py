import sys
import time
import numpy as np
 
from src_non_ros.mini_cheetah_tmotor_can.src.motor_driver.canmotorlib import CanMotorController

CAN_SOCKET = "can0"
CAN_MOTOR_ID = 0x4
 
    
motor = CanMotorController(CAN_SOCKET, CAN_MOTOR_ID, "AK10_9_V1p1")
motor.enable_motor()
 
pos, vel, curr = motor.send_deg_command(0, 0.0, 0.0, 0, 0.0)
 
time.sleep(2)
 
motor.disable_motor()
 