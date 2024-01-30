from pymycobot import Mercury
import time

left_arm = Mercury("/dev/ttyTHS0", debug=True)
right_arm = Mercury("/dev/ttyACM1", debug=True)

# right_arm.send_angles([0,10,0,-90,-90,90,0], 50)
# right_arm.send_base_coords([430.00, 31.00, 120.00, -179.93, 0.23, -0.32], 50)
# right_arm.set_servo_calibration(3)
# right_arm.send_base_coord(1, 326, 50)

right_arm.send_angle(1, 10, 10)