import typing as T
import numpy as np
from numpy.typing import NDArray
import cv2

MarkerPack = T.Dict[int, T.Tuple[NDArray, NDArray]]
OperationPoints = T.Dict[str, T.Tuple[float, float, float]]


def undistort_points(points, camera_matrix, dist_coeffs):
    """
    Apply distortion correction to a single point.

    :param point: The distorted point (x, y) in the image.
    :param camera_matrix: The camera intrinsic matrix.
    :param dist_coeffs: The distortion coefficients.
    :return: The undistorted point.
    """

    # Undistort the point
    dst = cv2.undistortPoints(points, camera_matrix, dist_coeffs, None, camera_matrix)

    return dst


def project_2d_to_3d(point, depth, camera_matrix):
    """
    Convert a 2D point in the image to a 3D point in camera coordinates.

    :param point: The 2D point (x, y) in the image.
    :param depth: The depth Z of the point.
    :param camera_matrix: The camera intrinsic matrix.
    :return: The 3D point (X, Y, Z) in camera coordinates.
    """
    # Intrinsic matrix decomposition
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Convert image coordinates to camera coordinates
    X = (point[0] - cx) * depth / fx
    Y = (point[1] - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z])


def project_3d_to_2d(point_3d, mtx, dist):
    """
    Project a 3D point in camera coordinates to 2D image coordinates considering lens distortion.

    :param point_3d: The 3D point (X, Y, Z) in camera coordinates.
    :param camera_matrix: The camera intrinsic matrix.
    :param dist_coeffs: The distortion coefficients.
    :return: The 2D image point (x, y).
    """
    # Convert point to required shape
    points_3d = np.array([point_3d], dtype=np.float32).reshape(-1, 1, 3)

    # Zero rotation and translation vectors
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    # Project 3D points to 2D image plane
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)
    points_2d = undistort_points(points_2d, mtx, dist)
    points_2d = points_2d.reshape((-1, 2))
    return points_2d


def calc_offset_points(
    base_vecs: T.Tuple[NDArray, NDArray, NDArray],
    base_point: NDArray,
    offsets: OperationPoints,
):
    res: OperationPoints = dict()
    vx, vy, vz = base_vecs
    for k, v in offsets.items():
        x, y, z = v
        p = base_point.copy()
        p += x * vx
        p += y * vy
        p += z * vz
        res[k] = tuple(p.squeeze())  # type: ignore
    return res


def plane_equation_from_points(p1, p2, p3) -> T.Tuple[float, float, float, float]:
    # Convert points to numpy arrays
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    p1 = p1.T
    p2 = p2.T
    p3 = p3.T
    # Calculate vectors AB and AC
    AB = p2 - p1
    AC = p3 - p1

    # Calculate the cross product (normal to the plane)
    N = np.cross(AB, AC)

    # Plane equation: Ax + By + Cz + D = 0
    # A, B, C are the components of the normal, D can be found using point p1
    D = -np.dot(N, p1.T)
    D = D.squeeze().item()

    N = -N.squeeze()
    D = D
    # stands for A, B, C, D
    return N[0], N[1], N[2], D


def pack_marker(ids: NDArray, rvecs: NDArray, tvecs: NDArray) -> MarkerPack:
    res: MarkerPack = {}
    for _id, rvec, tvec in zip(ids, rvecs, tvecs):
        _id = _id.item()
        res[_id] = (rvec, tvec)
    return res


def interpolation_z(
    A: float, B: float, C: float, D: float, x: float, y: float
) -> float:
    return -(A * x + B * y + D) / C


def normalize_vec(v: NDArray) -> NDArray:
    return v / np.linalg.norm(v)  # type: ignore


def get_base_vector(base, x_vec, y_vec):
    A, B, C, D = plane_equation_from_points(base, x_vec, y_vec)
    vz = np.array([A, B, C])
    vz = vz.reshape((3, 1))

    vx = x_vec - base
    vy = y_vec - base

    vx = normalize_vec(vx)
    vy = normalize_vec(vy)
    vz = normalize_vec(vz)

    return vx, vy, vz
