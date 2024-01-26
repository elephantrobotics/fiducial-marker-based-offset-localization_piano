import pyrealsense2 as rs  # type: ignore
import numpy as np
import time
import stag  # type: ignore
import cv2
import numpy as np
from marker_utils import ArucoDetector
import typing as T
from numpy.typing import NDArray

# Tested with camera D435


class RealSenseCamera:
    def __init__(
        self,
        capture_size: T.Tuple[int, int] = (640, 480),
        fps: int = 30,
        depth_cam: bool = False,
    ):
        super().__init__()
        self.capture_size = capture_size
        self.fps = fps
        self.depth_cam = depth_cam
        w, h = capture_size
        # Configure depth and color streams
        self.pipeline = rs.pipeline()  # type: ignore
        self.config = rs.config()  # type: ignore
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)  # type: ignore

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)  # type: ignore
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        sensor_infos = list(
            map(lambda x: x.get_info(rs.camera_info.name), device.sensors)  # type: ignore
        )

        if depth_cam:
            self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)  # type: ignore

    def capture(self):
        # Start streaming
        cfg = self.pipeline.start(self.config)
        profile = cfg.get_stream(rs.stream.color)  # type: ignore
        intr = profile.as_video_stream_profile().get_intrinsics()
        print(intr)
        # warm up
        for i in range(10):
            pipeline = self.pipeline
            frames = pipeline.wait_for_frames()

    def release(self):
        self.pipeline.stop()

    def update_frame(self) -> None:
        pipeline = self.pipeline
        frames = pipeline.wait_for_frames()
        self.curr_frame = frames
        self.curr_frame_time = time.time_ns()

    def color_frame(self) -> T.Union[np.ndarray, None]:
        frame = self.curr_frame.get_color_frame()
        if not frame:
            return None
        frame: NDArray = np.asanyarray(frame.get_data())  # type: ignore
        return frame

    def depth_frame(self) -> T.Union[np.ndarray, None]:
        frame = self.curr_frame.get_depth_frame()
        if not frame:
            return None
        frame: NDArray = np.asanyarray(frame.get_data())  # type: ignore
        return frame


# Test example
if __name__ == "__main__":
    import cv2

    cam = RealSenseCamera((1920, 1080), 30)
    cam.capture()
    # specify marker type
    libraryHD = 11
    mtx = np.array([[1367.86, 0, 937.803], [0, 1367.44, 563.887], [0, 0, 1]])
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    while True:
        cam.update_frame()

        color_frame = cam.color_frame()
        if color_frame is not None:
            cv2.imshow("Color", color_frame)

        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        elif k == ord("s"):
            cv2.imwrite("save.jpg", color_frame)  # type: ignore

    cam.release()
    cv2.destroyAllWindows()
