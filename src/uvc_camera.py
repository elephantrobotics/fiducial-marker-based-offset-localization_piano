import cv2
import numpy as np
from numpy.typing import NDArray, ArrayLike
import time
import typing as T


class UVCCamera:
    def __init__(
        self,
        cam_index=0,
        mtx=None,
        dist=None,
        capture_size: T.Tuple[int, int] = (640, 480),
        fps=30,
    ):
        super().__init__()
        self.cam_index = cam_index
        self.mtx = mtx
        self.dist = dist
        self.curr_color_frame: T.Union[np.ndarray, None] = None
        self.capture_size = capture_size
        self.fps = fps

    def capture(self) -> None:
        self.cap = cv2.VideoCapture(self.cam_index)
        width, height = self.capture_size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def update_frame(self) -> bool:
        ret, self.curr_color_frame = self.cap.read()
        ret: bool
        return ret

    def color_frame(self) -> T.Union[NDArray, None]:
        if self.curr_color_frame is not None:
            return self.curr_color_frame.copy()
        else:
            return None

    def release(self) -> None:
        self.cap.release()


if __name__ == "__main__":
    cam = UVCCamera(1)
    cam.capture()
    libraryHD = 11
    camera_params = np.load("camera_params_2d.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    while True:
        if not cam.update_frame():
            continue

        frame = cam.color_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        preview_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Color", preview_frame)
        if cv2.waitKey(1) == ord("q"):
            break
