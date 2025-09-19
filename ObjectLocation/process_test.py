from ultralytics import YOLO
import predict
import preprocess
import cv2
import tkinter as tk
import numpy as np
import location
import math
# 26mm
mtx = np.array([[711.71612643, 0, 849.68478682],
                [0, 712.4984464, 645.89697782],
                [0, 0, 1]])
# corner coordination 1.57 1.05 1.6
points3d = [[0, -3.6, 3.8],
            [0, 3.6, 3.8],
            [0, -3.6, -3.8],
            [0, 3.6, -3.8]]
data_path = "origin_img/measureddata/door numberIMG_2303(20230529-142703).JPG"

model = YOLO("E:/Yolo/ultralytics/runs/detect/train13/weights/best.pt")  # 加载预训练的 YOLOv8n 模型

image = cv2.imread(data_path)
# Predict with the model and initail
results = model(data_path)  # predict on an image
# 标签
lables = results[0].names
# 获取毛框
boxs = predict.getboxs(results, lables)[0]
keypoint = []
points_3d = points3d
for box in boxs:
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_local = preprocess.shear(img_gray, box)
    _, binary = cv2.threshold(img_local, 100, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opened, 25, 50)
    corners = cv2.goodFeaturesToTrack(edges, maxCorners=5, qualityLevel=0.01, minDistance=20)
    corners_sorted_x = sorted(corners, key=lambda corner: corner[0][0])
    # 提取最左边的角点作为左上角和左下角
    left_corners = sorted(corners_sorted_x[:2], key=lambda corner: corner[0][1])
    # 提取最右边的角点作为右上角和右下角
    right_corners = sorted(corners_sorted_x[-2:], key=lambda corner: corner[0][1])
    # 最终的四个角点
    top_left_corner = left_corners[0]
    bottom_left_corner = left_corners[1]
    top_right_corner = right_corners[0]
    bottom_right_corner = right_corners[1]
    corners = [top_left_corner, bottom_left_corner, top_right_corner, bottom_right_corner]
    for corner in corners:
        x, y = corner.ravel()
        point_r = preprocess.recover(image, box, [x, y])
        keypoint.append(point_r)
        cv2.circle(image, (int(point_r[0]), int(point_r[1])), radius=15, color=(0, 255, 0), thickness=-1)

h, w, _ = image.shape
root = tk.Tk()
screen_width = root.winfo_screenwidth()  # 获取屏幕的宽度
screen_height = root.winfo_screenheight()  # 获取屏幕的高度
scale = min(screen_width / w, screen_height / h)  # 计算缩放比例
img_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
x = int((screen_width - img_resized.shape[1]) / 2)
y = int((screen_height - img_resized.shape[0]) / 2)
cv2.imshow("Result", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

p1=points3d[0:4]
p2=keypoint[0:4]
# R, t, inliers = location.calculate_relative_position_ransac(mtx, p1, p2)
R, t = location.calculate_relative_position(mtx, points3d, keypoint)

# retval, rvec, t = cv2.solvePnP(points3d[0:3], keypoint[0:3], mtx, None, flags=cv2.SOLVEPNP_EPNP)

position = location.get_camera_origin_in_world_coord(R, t)
print(position)
# print(inliers)



h, w, _ = image.shape
# 显示图像

