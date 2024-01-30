import cv2
import numpy as np
from numpy.typing import NDArray, ArrayLike
import time
import typing as T
import stag  # type: ignore
from pathlib import Path
from uvc_camera import UVCCamera
import os
from marker_utils import solve_marker_pnp

if __name__ == "__main__":
    path_prefix = Path(os.path.dirname(__file__))
    cam = UVCCamera(0, capture_size=(2560, 1440))
    cam.capture()
    libraryHD = 11
    camera_params = np.load(str(path_prefix / Path("LRCP5020_params.npz")))
    mtx, dist = camera_params["mtx"], camera_params["dist"]
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
            stag.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs = solve_marker_pnp(corners, marker_size, mtx, dist)
            print(tvecs)
        preview_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Color", preview_frame)
        if cv2.waitKey(1) == ord("q"):
            break
