import numpy as np
import cv2


def calculate_relative_position(K, pts3d, pts2d):
    # K is the internal parameter matrix
    # pts3d are the 3D points of the object
    # pts2d are the corresponding 2D points in the image

    # Convert the points to numpy arrays
    pts3d = np.array(pts3d).astype(np.float64)
    pts2d = np.array(pts2d).astype(np.float64)

    # Use OpenCV's solvePnP function to calculate the relative position
    retval, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, None, flags=cv2.SOLVEPNP_EPNP)

    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec

def calculate_relative_position_ransac(K, pts3d, pts2d):
    # K is the internal parameter matrix
    # pts3d are the 3D points of the object
    # pts2d are the corresponding 2D points in the image

    # Convert the points to numpy arrays
    pts3d = np.array(pts3d).astype(np.float64)
    pts2d = np.array(pts2d).astype(np.float64)

    # Use OpenCV's solvePnP function to calculate the relative position
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, K, distCoeffs=None)

    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec, inliers

def get_camera_origin_in_world_coord(R_wc, t_wc):
    # 相机坐标系原点在世界坐标系中的坐标
    C_w = - R_wc.T @ t_wc

    return C_w
# Example usage
# K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]]) # Internal parameter matrix
# pts3d = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] # 3D points
# pts2d = [[500.0, 500.0], [600.0, 500.0], [500.0, 600.0], [500.0, 400.0]] # Corresponding 2D points
#
# R, t = calculate_relative_position(K, pts3d, pts2d)
# c = get_camera_origin_in_world_coord(R,t)
# print("Rotation matrix:")
# print(R)
# print("Translation vector:")
# print(t)
# print("Original coordinate")
# print(c)