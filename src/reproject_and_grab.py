import cv2
import numpy as np
from numpy.typing import NDArray, ArrayLike
import stag  # type: ignore
from marker_utils import solve_marker_pnp
import typing as T
from transformations import *
import os
import time
from collections import deque
from arm_control import *

from pymycobot import Mercury
from arm_control import *

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

MeanOperationPoints = T.Dict[str, T.Deque[T.Tuple[float, float, float]]]


def init_arm(arm):
    init_base_coords = [363.9, -90.4, 243.7, 180, 0.0, 0]
    init_angles = [11.79, 18.04, 0.0, -107.98, -83.68, 80.62, 65.45]
    # arm.send_base_coords(init_base_coords, 80)
    arm.send_angles(init_angles, 50)
    time.sleep(5)

    pump_off(arm)
    time.sleep(0.5)

    arm.set_tool_reference([0, 0, 70, 0, 0, 0])
    time.sleep(0.05)
    arm.set_end_type(1)


def calc_new_coords(arm_coords, tvecs) -> T.Union[np.ndarray, None]:
    # 单独算坐标
    if arm_coords is None or len(arm_coords) != 6:
        return None

    mat = homo_transform_matrix(*arm_coords) @ homo_transform_matrix(64.86, 0, -43, 0, 0, 180) @ homo_transform_matrix(-10, -10, 0, 0, 0, 0)
    # mat = homo_transform_matrix(*arm_coords) @ homo_transform_matrix(64.86, 0, -43, 0, 0, 180)
    rot = arm_coords[-3:]
    p_end = np.vstack([np.reshape(tvecs[0], (3, 1)), 1])
    p_base = np.squeeze((mat @ p_end)[:-1]).astype(int)
    new_coords = np.concatenate([p_base, rot])
    return new_coords


def calc_target_coords(arm_coords, tvec, rvec) -> T.Union[np.ndarray, None]:
    # 坐标+角度
    if arm_coords is None or len(arm_coords) != 6:
        return None
    tvec = tvec.squeeze().tolist()
    rvec = rvec.squeeze().tolist()
    mat = (
        homo_transform_matrix(*arm_coords)
        @ homo_transform_matrix(0, -70, 0, 0, 0, 90)
        @ homo_transform_matrix(*tvec, *rvec)
    )
    rot_mat = mat[:3, :3]
    r: np.ndarray = cvt_rotation_matrix_to_euler_angle(rot_mat)
    r = r * 180 / np.pi
    t: np.ndarray = mat[:3, 3]
    res = np.hstack([t.squeeze(), r.squeeze()])
    return res


def calc_xy_move_coords(
    arm_coords: np.ndarray, target_move: np.ndarray
) -> T.Union[np.ndarray, None]:
    z_now = arm_coords[2]
    new_coords = target_move.copy()
    new_coords[2] = z_now
    return new_coords


def draw_frame(frame, base_vecs, point_3d, mtx, dist):
    vx, vy, vz = base_vecs
    bx, by = project_3d_to_2d(point_3d, mtx, dist).squeeze().astype(np.int32)

    px = point_3d.copy()
    px += vx * 50
    x_vx, y_vx = project_3d_to_2d(px, mtx, dist).squeeze().astype(np.int32)

    py = point_3d.copy()
    py += vy * 50
    x_vy, y_vy = project_3d_to_2d(py, mtx, dist).squeeze().astype(np.int32)

    pz = point_3d.copy()
    pz += vz * 50
    x_vz, y_vz = project_3d_to_2d(pz, mtx, dist).squeeze().astype(np.int32)

    cv2.arrowedLine(frame, (bx, by), (x_vx, y_vx), (0, 0, 255), 3)
    cv2.arrowedLine(frame, (bx, by), (x_vy, y_vy), (0, 255, 0), 3)
    cv2.arrowedLine(frame, (bx, by), (x_vz, y_vz), (255, 0, 0), 3)


def draw_grid(frame, base_vecs, point_3d, mtx, dist):
    vx, vy, vz = base_vecs

    for i in range(10 + 1):
        for j in range(5 + 1):
            p = point_3d.copy()
            p += i * vx * 20
            p += j * vy * 20
            x, y = project_3d_to_2d(p, mtx, dist).squeeze().astype(np.int32)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), 3, cv2.FILLED)


def read_offset_table(filename: str) -> OperationPoints:
    res: OperationPoints = {}
    with open(filename, "r") as f:
        t = f.read()
        t = t.strip()
        lines = t.split("\n")
        content = lines[1:]
        for line in content:
            items = line.split(",")
            name, x, y, z = items
            res[name] = (float(x), float(y), float(z))
    return res


def draw_points(
    frame: NDArray,
    points: OperationPoints,
    mtx: NDArray,
    dist: NDArray,
    color=(0, 0, 255),
):
    for k, v in points.items():
        x, y = project_3d_to_2d(v, mtx, dist).squeeze().astype(np.int32)
        cv2.circle(frame, (x, y), 3, color, 3, cv2.FILLED)


def draw_texts(
    frame: NDArray,
    points: OperationPoints,
    mtx: NDArray,
    dist: NDArray,
    color=(0, 0, 255),
):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    thickness = 1
    for k, v in points.items():
        x, y = project_3d_to_2d(v, mtx, dist).squeeze().astype(np.int32)
        ((w, h), length) = cv2.getTextSize(k, font, font_scale, thickness)
        x -= w // 2
        cv2.putText(frame, k, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)
        # cv2.circle(frame, (x, y), 3, color, 3, cv2.FILLED)


def init_op_points(offset_table, max_size=50) -> MeanOperationPoints:
    res: MeanOperationPoints = dict()
    for k, v in offset_table.items():
        res[k] = deque(maxlen=max_size)
    return res


def update_mean_op_points(
    mean_op_points: MeanOperationPoints, op_points: OperationPoints
):
    for k, v in op_points.items():
        mean_op_points[k].append(v)


def get_mean_op_points(mean_op_points: MeanOperationPoints) -> OperationPoints:
    res: OperationPoints = dict()
    for k, v in mean_op_points.items():
        res[k] = np.mean(np.array(v), axis=0)  # type: ignore
    return res


from uvc_camera import UVCCamera
from pathlib import Path

path_prefix = Path(os.path.dirname(__file__))

cam = UVCCamera(0, capture_size=(2560, 1440), fps=30)
cam.capture()

params = np.load(str(path_prefix / Path("LRCP5020_params.npz")))
mtx, dist = params["mtx"], params["dist"]
libraryHD = 11

csv_file = path_prefix / Path("cad_points.csv")
offsets = read_offset_table(str(csv_file))
mean_op_points = init_op_points(offsets, max_size=10)

right_arm = Mercury("/dev/ttyACM1")

right_arm.set_tool_reference([0,0,59,0,0,0])
time.sleep(0.03)
right_arm.set_end_type(1)
time.sleep(0.03)

right_arm.send_angles([0,10,0,-90,-90,90,0], 50)
time.sleep(3)

# right_arm.send_base_coords([283.4, -0.1, 251.0, -179.72, -0.27, 94.0], 50)
# time.sleep(5)

# right_arm.send_angles([10.02, 34.28, 0.0, -117.17, -80.73, 84.05, -0.11], 50)
# time.sleep(3)

right_arm.send_angles([20.21, 36.55, 0.0, -112.5, -72.03, 78.54, -1.29], 50)
time.sleep(3)

for i in range(40):
    # frame = cv2.imread("save.jpg")
    cam.update_frame()
    if i < 20:
        time.sleep(0.1)
        continue

    frame = cam.color_frame()
    if frame is None:
        time.sleep(0.1)
        continue

    # frame = cv2.flip(frame, -1)
    # cv2.imshow("preview", frame)
    # if cv2.waitKey(1) == ord("q"):
        # break

    (corners, ids, rejected_corners) = stag.detectMarkers(frame, libraryHD)  # type: ignore
    if len(ids) != 3:
        time.sleep(0.1)
        continue

    rvecs, tvecs = solve_marker_pnp(corners, 12, mtx, dist)

    pack = pack_marker(ids, rvecs, tvecs)
    tag0, tag1, tag2 = pack[0][1], pack[1][1], pack[2][1]
    print(tag0, tag1, tag2)

    vx, vy, vz = get_base_vector(tag0, tag1, tag2)
    print(f"vx:{vx.squeeze()} vy:{vy.squeeze()} vz:{vz.squeeze()}")

    draw_frame(frame, (vx, vy, vz), pack[0][1], mtx, dist)

    raw_op_points = calc_offset_points((vx, vy, vz), tag0, offsets)
    update_mean_op_points(mean_op_points, raw_op_points)
    op_points = get_mean_op_points(mean_op_points)

    # draw_points(frame, op_points, mtx, dist)
    draw_texts(frame, op_points, mtx, dist)

    points = project_3d_to_2d(tvecs, mtx, dist)
    points = points.astype(np.int32)
    for p in points:
        cv2.circle(frame, p, 5, (0, 0, 255), 3)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord("q"):
        break


op_points = get_mean_op_points(mean_op_points)
arm_base_coords = get_base_coords(right_arm)

for c in "UIOPZXCV":
    cam_coords = op_points[c]
    new_base_coords = calc_new_coords(arm_base_coords, [cam_coords])
    new_base_coords[2] -= 6
    xy_coords = new_base_coords.copy()
    xy_coords[2] += 50
    print(new_base_coords)
    right_arm.send_base_coords(xy_coords, 50)
    time.sleep(4)
    right_arm.send_base_coords(new_base_coords, 30)
    right_arm.send_base_coords(xy_coords, 50)