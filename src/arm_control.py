from pymycobot import CommandGenerator, Mercury  # type: ignore
import typing as T
import time

def get_base_coords(arm : Mercury):
    coords = None
    for _ in range(5):
        coords = arm.get_base_coords()
        if coords is not None and len(coords) != 0:
            break
    
    return coords

def move_x_increment(arm: CommandGenerator, val: float):
    coords: T.Any = arm.get_coords()
    coords[0] += val
    arm.send_coords(coords, 10)


def move_y_increment(arm: CommandGenerator, val: float):
    coords: T.Any = arm.get_coords()
    coords[1] += val
    arm.send_coords(coords, 10)


def move_z_increment(arm: CommandGenerator, val: float):
    coords: T.Any = arm.get_coords()
    coords[2] += val
    arm.send_coords(coords, 10)


# 开启吸泵
def pump_on(arm):
    arm.set_digital_output(1, 0)
    arm.set_digital_output(2, 1)


# 关闭吸泵
def pump_off(arm):
    arm.set_digital_output(1, 1)
    arm.set_digital_output(2, 0)
    time.sleep(0.05)
    arm.set_digital_output(1, 0)
