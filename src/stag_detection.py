import cv2
import numpy as np
from numpy.typing import NDArray, ArrayLike
import time
import typing as T
from marker_utils import solve_marker_pnp, draw_marker
import stag  # type: ignore
from queue import Queue
from uvc_camera import UVCCamera


def interp(pos0: np.ndarray, pos1: np.ndarray, step: int):
    x_step = (pos1[0] - pos0[0]) / step
    y_step = (pos1[1] - pos0[1]) / step
    z_step = (pos1[2] - pos0[2]) / step
    pos_step = np.array((x_step, y_step, z_step))
    res = []
    curr = pos0.copy()
    for i in range(step):
        curr += i * pos_step
        res.append(curr.copy())

    print(pos0)
    print(pos1)
    print(res)

    return res


def draw_3d_points_to_frame(
    obj_points: NDArray,
    frame: NDArray,
    rvec: NDArray,
    tvec: NDArray,
    mtx: NDArray,
    dist: NDArray,
) -> NDArray:
    image_points, _ = cv2.projectPoints(
        objectPoints=obj_points,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=mtx,
        distCoeffs=dist,
    )

    # 获取投影后的2D坐标
    image_points = image_points.squeeze()
    for x, y in image_points:
        x = int(x)
        y = int(y)
        cv2.circle(frame, (x, y), 2, color=(255, 0, 0), thickness=3)

    return frame


if __name__ == "__main__":
    cam = UVCCamera(0, capture_size=(2560, 1440))
    cam.capture()
    libraryHD = 11
    camera_params = np.load("LRCP5020_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    rvec_queue: Queue[NDArray] = Queue(100)
    marker_size = 12
    while True:
        if not cam.update_frame():
            continue

        frame: T.Union[NDArray, None] = cam.color_frame()  # type: ignore
        if frame is None:
            time.sleep(0.01)
            continue
        else:
            # detect markers
            (corners, ids, rejected_corners) = stag.detectMarkers(frame, libraryHD)  # type: ignore
            if len(ids) == 1:
                rvecs, tvecs = solve_marker_pnp(corners, marker_size, mtx, dist)
                rvec_queue.put(rvecs[0])
                rvec = np.mean(np.array(rvec_queue.queue), axis=0)

                obj_points = np.array(
                    [
                        # [40, -5, 0],
                        # [40, -25, 0],
                        # [40, -45, 0],
                        # [40, 15, 0],
                        # [40, 35, 0],
                        # [40, 55, 0],
                        # [20, 5, 0],
                        # [20, -15, 0],
                        # [20, 25, 0],
                        # [20, 45, 0],
                        # [20, -35, 0],
                        # [20, -55, 0],
                        [0, -20, 0],
                        [0, 20, 0],
                        [0, 40, 0],
                        [0, 60, 0],
                        [0, -40, 0],
                        [0, -60, 0],
                        # [-20, 10, 0],
                        # [-20, 30, 0],
                        # [-20, 50, 0],
                        # [-20, -10, 0],
                        # [-20, -30, 0],
                        # [-20, -50, 0],
                    ],
                    dtype=np.float64,
                )

                frame = draw_3d_points_to_frame(
                    obj_points, frame, rvecs[0], tvecs[0], mtx, dist
                )
                if frame is not None:
                    draw_marker(frame, corners, tvecs, rvecs, ids, mtx, dist)

        preview_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Color", preview_frame)
        if cv2.waitKey(1) == ord("q"):
            break
