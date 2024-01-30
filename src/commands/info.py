from pymycobot import Mercury
import numpy as np
import time

left_arm = Mercury("/dev/ttyTHS0")
right_arm = Mercury("/dev/ttyACM1")

np.set_printoptions(suppress=True)

print(f"Left angles : {left_arm.get_angles()}")
time.sleep(0.03)
print(f"Right angles : {right_arm.get_angles()}")
time.sleep(0.03)
print(f"Left coords : {left_arm.get_coords()}")
time.sleep(0.03)
print(f"Right coords : {right_arm.get_coords()}")
time.sleep(0.03)
print(f"Left base coords : {left_arm.get_base_coords()}")
time.sleep(0.03)
print(f"Right base coords : {right_arm.get_base_coords()}")
time.sleep(0.03)