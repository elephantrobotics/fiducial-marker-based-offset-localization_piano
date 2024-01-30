from pymycobot import Mercury
import time

left_arm = Mercury("/dev/ttyTHS0", debug=True)
right_arm = Mercury("/dev/ttyACM1", debug=True)

arm = right_arm
now_base_coords = arm.get_base_coords()
x,y,z,rx,ry,rz = now_base_coords
x += 0
y += 0
z += 50
# rx = 180
# ry = 0
# rz = 0
arm.send_base_coords([x,y,z,rx,ry,rz], 50)