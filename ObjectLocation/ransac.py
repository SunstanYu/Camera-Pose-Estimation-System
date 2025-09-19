import cv2
import numpy as np
import time

start_time = time.time()
# 生成随机的三维点和对应的二维点
object_points = np.random.rand(6, 3)  # 6个三维点
image_points = np.random.rand(6, 2)  # 6个二维点

# 定义相机内参
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]).astype(float)

# 使用RANSAC算法估计相机位姿
retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, distCoeffs=None)
end_time = time.time()

# 计算运行时间（以秒为单位）
run_time = end_time - start_time

# 打印运行时间
print("程序运行时间：", run_time, "秒")
start_time1 = time.time()
object_points = np.random.rand(30, 3)  # 6个三维点
image_points = np.random.rand(30, 2)  # 6个二维点

# 定义相机内参
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]).astype(float)

# 使用RANSAC算法估计相机位姿
retval1, rvec1, tvec1, inliers1 = cv2.solvePnPRansac(object_points, image_points, camera_matrix, distCoeffs=None)
end_time1 = time.time()
print(inliers1)
# 计算运行时间（以秒为单位）
run_time1 = end_time1 - start_time1

# 打印运行时间
print("程序运行时间：", run_time1, "秒")
# # 打印结果
# print("Rotation Vector:")
# print(rvec)
# print("Translation Vector:")
# print(tvec)
#
# # 将旋转向量转换为旋转矩阵
# rotation_matrix, _ = cv2.Rodrigues(rvec)
#
# # 打印旋转矩阵
# print("Rotation Matrix:")
# print(rotation_matrix)