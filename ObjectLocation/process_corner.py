from ultralytics import YOLO
import predict
import preprocess
import cv2
import tkinter as tk
import numpy as np
import location
import corner
import SIFT
import display
import time
import canny

method = 0


def points_generator(type, position, size):
    points_final = []
    w = size[0]
    h = size[1]
    if type == 'cup':
        points3d = [(0, 0, 0),
                    (7.9, 0, 0),
                    (0, 4.8, 0),
                    (7.9, 4.8, 0)]
        for p in position:
            x = int(p[0])
            y = int(p[1])
            point = (points3d[1][0] * x / w, points3d[2][1] * (h - y) / h, 0)
            points_final.append(point)
    if type == 'card':
        points3d = [(0, 0, 0),
                    (15.4, 0, 0),
                    (0, 7.5, 0),
                    (15.4, 7.5, 0)]
        for p in position:
            x = int(p[0])
            y = int(p[1])
            point = ((points3d[1][0]) * x / w, (points3d[2][1]) * (h - y) / h, 0)
            points_final.append(point)
    return points_final


def draw_points_and_lines(image, coordinates, c=None):
    coordinates = np.array(coordinates).astype(int)
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算图像底边中点的坐标
    mid_x, mid_y = int(width / 2), height - 1

    # 循环遍历坐标列表并在图像上绘制点和线
    if c is not None:
        cv2.putText(image, f'({c[0]},{c[1]},{c[2]})', (mid_x - 30, mid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0),
                    1)
    for point in coordinates:
        x = int(point[0])
        y = int(point[1])
        # 绘制点
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        # 绘制线
        cv2.line(image, (x, y), (mid_x, mid_y), (0, 255, 0), 2)
    # 返回绘制完成的图像
    return image


# 26mm
mtx = np.array([[751.73392551, 0, 753.85550879],
                [0, 753.8908959, 732.11882348],
                [0, 0, 1]])
# corner coordination 1.57 1.05 1.6
points3d = [(0, 0, 0),
            (15.4, 0, 0),
            (0, 7.2, 0),
            (15.4, 7.2, 0)]
data_path = "origin_img/measureddata/img5.jpg"
template = cv2.imread('template/card1.jpg.jpg', 0)
lable_path = "origin_img/measureddata/data/data5.txt"

image = cv2.imread(data_path)
image_ori = image.copy()
h, w, _ = image.shape
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
size_o = img_gray.shape[::-1]

keypoint = []
points_3d = points3d
n = 0

# Predict with the model and initail
# model = YOLO("E:/Yolo/ultralytics/runs/detect/train13/weights/best.pt")  # 加载预训练的 YOLOv8n 模型
# results = model(data_path)  # predict on an image
# lable_path = results[0].names
# boxs = predict.getboxs(results, lable_path)[0]

with open(lable_path, "r") as f:
    if f.readable():
        # 文件以可读模式打开，可以进行读取操作
        file = f.readlines()
        objects = []
        for i in file:
            objects.append(i.split(' '))
        objects = objects[0]
        x = round(np.float64(objects[1]) * size_o[0])
        y = round(np.float64(objects[2]) * size_o[1])
        width = round(np.float64(objects[3]) * size_o[0])
        height = round(np.float64(objects[4]) * size_o[1])
        boxs = [[int(x - width / 2), int(y - height / 2), (x + width / 2), (y + height / 2)]]

start_time = time.time()
for box in boxs:
    if n != 0:
        points3d.append(points_3d[0])
        points3d.append(points_3d[1])
        points3d.append(points_3d[2])
        points3d.append(points_3d[3])
    n = n + 1
    keypoint.append((int(box[0]), int(box[3])))
    keypoint.append((int(box[2]), int(box[3])))
    keypoint.append((int(box[0]), int(box[1])))
    keypoint.append((int(box[2]), int(box[1])))

    if method == 0:

        mask = preprocess.mask_generate(img_gray, box)
        _, binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(opened, 25, 50)
        # corners = corner.my_good_featuressToTrack(edges, maxCorners=30, qualityLevel=0.01, minDistance=30, mask=mask)

        corners = cv2.goodFeaturesToTrack(edges, maxCorners=30, qualityLevel=0.01, minDistance=30, mask=mask)

        corners_sorted_x = sorted(corners, key=lambda corner: corner[0][0])
        # cv2.imshow("Result", edges)

        # 提取最左边的角点作为左上角和左下角
        left_corners = sorted(corners_sorted_x[:2], key=lambda corner: corner[0][1])
        # 提取最右边的角点作为右上角和右下角
        right_corners = sorted(corners_sorted_x[-2:], key=lambda corner: corner[0][1])
        # 最终的四个角点
        top_left_corner = tuple(left_corners[0][0])
        bottom_left_corner = tuple(left_corners[1][0])
        top_right_corner = tuple(right_corners[0][0])
        bottom_right_corner = tuple(right_corners[1][0])
        corners = [bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner]
        print(corners)
        for point in corners:
            x = point[0]
            y = point[1]
            cv2.circle(edges, (int(x), int(y)), radius=5, color=(255, 255, 255), thickness=-1)
        # display.show_img(edges)
        if [corners]:
            if preprocess.is_approx_rectangle(corners):
                keypoint = corners
    elif method == 1:
        mask = preprocess.mask_generate_temp(img_gray, box)
        # display.show_img(mask)
        size = template.shape[::-1]
        img_s, points2d, p2 = SIFT.sift_match_images(mask, template)
        img_out = draw_points_and_lines(image, points2d)
        points3d = points_generator('card', p2, size)
        if len(points2d) >= 4:
            keypoint = points2d
        # save_path = "result/single_object/img8.jpg"
        # if save_path is not None:
        #     cv2.imwrite(save_path, img_s)
        display.show_img(img_s)

if len(keypoint) == 4:
    R, t = location.calculate_relative_position(mtx, points3d, keypoint)
else:
    R, t, inliers = location.calculate_relative_position_ransac(mtx, points3d, keypoint)

# retval, rvec, t = cv2.solvePnP(points3d[0:3], keypoint[0:3], mtx, None, flags=cv2.SOLVEPNP_EPNP)

position = location.get_camera_origin_in_world_coord(R, t)
endtime = time.time()
print(f"time: {endtime - start_time}")
print(position)
# print(inliers)

for point in keypoint:
    x = point[0]
    y = point[1]
    cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
img_out = draw_points_and_lines(image, keypoint, position)
# save_path = "result/single_object/img9.jpg"
# if save_path is not None:
#     cv2.imwrite(save_path, img_out)

# 显示图像
display.show_img(img_out)
