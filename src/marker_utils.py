import cv2
import numpy as np
import typing as T
from numpy.typing import NDArray, ArrayLike


class MarkerInfo(T.TypedDict):
    corners: np.ndarray
    tvec: np.ndarray
    rvec: np.ndarray
    num_id: int


def solve_marker_pnp(corners: NDArray, marker_size: int, mtx: NDArray, dist: NDArray):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    rvecs = []
    tvecs = []
    for corner in corners:
        retval, rvec, tvec = cv2.solvePnP(
            marker_points,
            corner,
            mtx,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if retval:
            rvecs.append(rvec)
            tvecs.append(tvec)

    rvecs = np.array(rvecs)  # type: ignore
    tvecs = np.array(tvecs)  # type: ignore
    (rvecs - tvecs).any()  # type: ignore
    return rvecs, tvecs


def draw_marker(frame: np.ndarray, corners, tvecs, rvecs, ids, mtx, dist) -> None:
    # cv2.aruco.drawDetectedMarkers(frame, corners, None, borderColor=(0, 255, 0))
    cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 200, 200))
    for i in range(len(ids)):
        corner, tvec, rvec, marker_id = corners[i], tvecs[i], rvecs[i], ids[i]
        cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 60, 2)


# deprecated
class ArucoDetector:
    def __init__(self, mtx: np.ndarray, dist: np.ndarray, marker_size: float):
        self.mtx = mtx
        self.dist = dist
        self.marker_size = marker_size
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        # parameters.useAruco3Detection = True
        # parameters.minMarkerDistanceRate = 0.03
        # parameters.minDistanceToBorder = 1
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    def estimatePoseSingleMarkers(self, corners):
        """
        This will estimate the rvec and tvec for each of the marker corners detected by:
           corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        """
        marker_points = np.array(
            [
                [-self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, -self.marker_size / 2, 0],
                [-self.marker_size / 2, -self.marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        rvecs = []
        tvecs = []
        for corner in corners:
            retval, rvec, tvec = cv2.solvePnP(
                marker_points,
                corner,
                self.mtx,
                self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if retval:
                rvecs.append(rvec)
                tvecs.append(tvec)

        rvecs = np.array(rvecs)
        tvecs = np.array(tvecs)
        (rvecs - tvecs).any()
        return rvecs, tvecs

    def detect_marker_corners(
        self, frame: np.ndarray
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        return corners, ids, rejectedImgPoints  # type: ignore

    def draw_marker(self, frame: np.ndarray, corners, tvecs, rvecs, ids) -> None:
        # cv2.aruco.drawDetectedMarkers(frame, corners, None, borderColor=(0, 255, 0))
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 200, 200))
        for i in range(len(ids)):
            corner, tvec, rvec, marker_id = corners[i], tvecs[i], rvecs[i], ids[i]
            cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec, tvec, 60, 2)

    @classmethod
    def draw_real_position_info(cls, frame: np.ndarray, corners, tvecs, trans_mat):
        n = len(corners)
        for i in range(n):
            corner, tvec = corners[i], tvecs[i]
            center = np.mean(corner, axis=1).squeeze().astype(int)
            p_end = np.vstack([np.reshape(tvec, (3, 1)), 1])
            p_base = np.squeeze((trans_mat @ p_end)[:-1]).astype(int)
            x, y, z = p_base
            font_size = 0.4
            cv2.putText(
                frame,
                f"x:{x}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"y:{y}",
                center + (0, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"z:{z}",
                center + (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    @classmethod
    def draw_position_info(cls, frame: np.ndarray, corners, tvecs):
        n = len(corners)
        for i in range(n):
            corner, tvec = corners[i], tvecs[i]
            center = np.mean(corner, axis=1).squeeze().astype(int)
            x, y, z = np.squeeze(tvec).astype(int)
            cv2.putText(
                frame,
                f"x:{x}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"y:{y}",
                center + (0, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"z:{z}",
                center + (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    @classmethod
    def console_view_marker_pos3d(cls, data: T.List[MarkerInfo]):
        data = data.copy()
        data.sort(key=lambda k: k["num_id"])
        for obj in data:
            num_id = obj["num_id"]
            tvec = obj["tvec"]
            x, y, z = np.squeeze(tvec).astype(int)
            print(f"id:{num_id}")
            print(f"x:{x} y:{y} z:{z}")
        print()

    @classmethod
    def make_structure_data(cls, corners, ids, rvecs, tvecs) -> T.List[MarkerInfo]:
        data = []
        n = len(corners)
        for i in range(n):
            corner, n_id, rvec, tvec = corners[i], ids[i][0], rvecs[i], tvecs[i]
            corner = np.squeeze(corner)
            rvec = np.squeeze(rvec)
            tvec = np.squeeze(tvec)
            obj = MarkerInfo(corners=corner, tvec=tvec, rvec=rvec, num_id=n_id)
            data.append(obj)
        return data
