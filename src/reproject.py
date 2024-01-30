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

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

MeanOperationPoints = T.Dict[str, T.Deque[T.Tuple[float, float, float]]]


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


def read_offset_table(filename: str):
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


# cam = RealSenseCamera(capture_size=(1920, 1080), fps=30)
# cam.capture()
# mtx = np.array([[1367.86, 0, 937.803], [0, 1367.44, 563.887], [0, 0, 1]])
# dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

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

while True:
    # frame = cv2.imread("save.jpg")
    cam.update_frame()
    frame = cam.color_frame()
    if frame is None:
        time.sleep(0.1)
        continue
    frame = cv2.flip(frame, -1)
    cv2.imshow("preview", frame)
    if cv2.waitKey(1) == ord("q"):
        break

    (corners, ids, rejected_corners) = stag.detectMarkers(frame, libraryHD)  # type: ignore
    if len(ids) != 3:
        time.sleep(0.1)
        continue

    rvecs, tvecs = solve_marker_pnp(corners, 12, mtx, dist)

    pack = pack_marker(ids, rvecs, tvecs)
    tag0, tag1, tag2 = pack[0][1], pack[1][1], pack[2][1]
    vx, vy, vz = get_base_vector(tag0, tag1, tag2)
    print(f"vx:{vx.squeeze()} vy:{vy.squeeze()} vz:{vz.squeeze()}")

    draw_frame(frame, (vx, vy, vz), pack[0][1], mtx, dist)

    raw_op_points = calc_offset_points((vx, vy, vz), tag0, offsets)
    update_mean_op_points(mean_op_points, raw_op_points)
    op_points = get_mean_op_points(mean_op_points)
    # draw_points(frame, op_points, mtx, dist)
    draw_texts(frame, op_points, mtx, dist)  # type: ignore

    points = project_3d_to_2d(tvecs, mtx, dist)
    points = points.astype(np.int32)
    for p in points:
        cv2.circle(frame, p, 5, (0, 0, 255), 3)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord("q"):
        break
