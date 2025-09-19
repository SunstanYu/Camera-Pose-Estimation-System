import cv2

import SIFT
import display
import location as lc
import numpy as np
import tkinter as tk


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
            point = ((points3d[1][0]) * x / w, (points3d[2][1]) * y / h, 0)
            points_final.append(point)
    return points_final


def single_object(arr, size):
    arr = arr[0]
    cla = arr[0]
    x = round(np.float64(arr[1]) * size[0])
    y = round(np.float64(arr[2]) * size[1])
    width = round(np.float64(arr[3]) * size[0])
    height = round(np.float64(arr[4]) * size[1])

    points2d = [(x - width / 2, y + height / 2),
                (x + width / 2, y + height / 2),
                (x - width / 2, y - height / 2),
                (x + width / 2, y - height / 2)]
    # points2d = [(x - width / 2, y - height / 2),
    #             (x + width / 2, y - height / 2),
    #             (x - width / 2, y + height / 2),
    #             (x + width / 2, y + height / 2)]
    points3d = points_generator('cup', points2d, size)
    # print(points3d)
    # points2d = [(x, y + height / 2),
    #             (x, y + height / 6),
    #             (x, y - height / 6),
    #             (x, y - height / 2)]
    return points2d, points3d


def double_object(arr, size):
    cla = []
    x = []
    y = []
    width = []
    height = []
    for i in range(0, 2):
        cla.append(arr[i][0])
        x.append(round(np.float64(arr[i][1]) * size[0]))
        y.append(round(np.float64(arr[i][2]) * size[1]))
        width.append(round(np.float64(arr[i][3]) * size[0]))
        height.append(round(np.float64(arr[i][4]) * size[1]))
    points2d = [(x[0], y[0] - height[0] / 6),
                (x[0], y[0] + height[0] / 6),
                (x[1], y[1] - height[1] / 6),
                (x[1], y[1] + height[1] / 6)]
    return points2d


def quadruple_object(arr, size):
    cla = []
    x = []
    y = []
    width = []
    height = []
    for i in range(0, 4):
        cla.append(arr[i][0])
        x.append(round(np.float64(arr[i][1]) * size[0]))
        y.append(round(np.float64(arr[i][2]) * size[1]))
        width.append(round(np.float64(arr[i][3]) * size[0]))
        height.append(round(np.float64(arr[i][4]) * size[1]))
    points2d = [(x[0], y[0]),
                (x[1], y[1]),
                (x[2], y[2]),
                (x[3], y[3])]
    return points2d


def draw_points_and_lines(image, coordinates, c=None):
    coordinates = np.array(coordinates).astype(int)
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算图像底边中点的坐标
    mid_x, mid_y = int(width / 2), height - 1

    # 循环遍历坐标列表并在图像上绘制点和线
    if [c]:
        cv2.putText(image, f'({c[0]},{c[1]},{c[2]})', (mid_x - 30, mid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0),
                    1)
    for x, y in coordinates:
        # 绘制点
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        # 绘制线
        cv2.line(image, (x, y), (mid_x, mid_y), (0, 255, 0), 2)

    # 返回绘制完成的图像
    return image


if __name__ == '__main__':
    # single
    img = cv2.imread("E:/Yolo/ultralytics/runs/detect/predict5/img3.jpg")
    # double
    # img = cv2.imread("chair_data/predict/image/img1.jpg")
    # quadruple
    # img = cv2.imread("chair_data/predict/image/img5.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    # single
    path = "E:/Yolo/ultralytics/runs/detect/predict5/labels/img3.txt"
    path_temp = "template/cup.jpg"
    # double
    # path = "chair_data/predict/label/img1.txt"
    # quadruple
    # path = "chair_data/predict/label/img5.txt"
    template = cv2.imread(path_temp)
    gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    size_temp = gray_temp.shape[::-1]
    match_img, points2d, points_B = SIFT.sift_match_images(gray, gray_temp)
    display.show_img(match_img)
    points3d = points_generator('cup', points_B, size_temp)
    # with open(path, "r") as f:
    #     if f.readable():
    #         # 文件以可读模式打开，可以进行读取操作
    #         location = f.readlines()
    #         objects = []
    #         for i in location:
    #             objects.append(i.split(' '))
    #         if len(objects) == 1:
    #             points2d, points3d = single_object(objects, size)
    #         elif len(objects) == 2:
    #             points2d = double_object(objects, size)
    #         elif len(objects) == 4:
    #             points2d = quadruple_object(objects, size)
    #     else:
    #         # 文件不支持读取操作，可能是以只写模式打开
    #         print("Error: File is not readable.")
    # single object
    # points3d = [(0, 6.95, 0),
    #             (0, -3.95, 0),
    #             (0, 6.95, 4.8),
    #             (0, -3.95, 4.8)]
    # points3d = [(0, 0, 0),
    #             (7.9, 0, 0),
    #             (0, 4.8, 0),
    #             (7.9, 4.8, 0)]
    # points3d = [(0, 0, 4.8),
    #             (0, 0, 3.2),
    #             (0, 0, 1.6),
    #             (0, 0, 0)]
    # double objects
    # points3d = [[0, 0, 28.833],
    #             [0, 0, 57.667],
    #             [0, 102.5, 28.833],
    #             [0, 102.5, 57.667]]
    # quadruple objects
    # points3d = [[80, -78.9, 43.25],
    #             [0, 0, 43.25],
    #             [0, 102.5, 43.25],
    #             [96.4, 132.6, 43.25]]
    # f= 41.6mm
    # mtx = np.array([[2.22814246e+03, 0.00000000e+00, 8.66009438e+02],
    #                 [0.00000000e+00, 2.22770590e+03, 6.62392340e+02],
    #                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # f= 26 mm
    mtx = np.array([[751.73392551, 0, 753.85550879],
                    [0, 753.8908959, 732.11882348],
                    [0, 0, 1]])
    # f= 13mm
    # mtx = np.array([[711.71612643, 0, 849.68478682],
    #                 [0, 712.4984464, 645.89697782],
    #                 [0, 0, 1]])
    R, t, inlines = lc.calculate_relative_position_ransac(mtx, points3d, points2d)
    c = lc.get_camera_origin_in_world_coord(R, t)
    print("Rotation matrix:")
    print(R)
    print("Translation vector:")
    print(t)
    print("Original coordinate")
    print(c)
    img_out = draw_points_and_lines(img, points2d, c)
    h, w, _ = img_out.shape
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()  # 获取屏幕的宽度
    screen_height = root.winfo_screenheight()  # 获取屏幕的高度

    scale = min(screen_width / w, screen_height / h)  # 计算缩放比例
    img_resized = cv2.resize(img_out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x = int((screen_width - img_resized.shape[1]) / 2)
    y = int((screen_height - img_resized.shape[0]) / 2)
    # single
    # save_path = "result/single_object/img2.jpg"
    # double
    save_path = "result/double_object/img1.jpg"
    # quadruple
    # save_path = "result/quadruple_object/img1.jpg"
    if save_path is not None:
        cv2.imwrite(save_path, img_resized)
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("output", 800, 600)  # 设置窗口大小
    # cv2.moveWindow("output", 400, 200)
    cv2.imshow("output", img_resized)
    cv2.moveWindow('output', x, y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
