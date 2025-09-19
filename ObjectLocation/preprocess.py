import numpy as np
import cv2
import math


def shear(image, area):
    h, w = image.shape
    # 定义目标区域的坐标（左上角和右下角）
    x1 = max(int(area[0]) - 25, 0)
    y1 = max(int(area[1]) - 25, 0)
    x2 = min(int(area[2]) + 25, w)
    y2 = min(int(area[3]) + 25, h)

    # 从图像中剪切出目标区域
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


def recover(image, area, point):
    h, w, _ = image.shape
    x1 = max(int(area[0]) - 25, 0)
    y1 = max(int(area[1]) - 25, 0)
    x2 = min(int(area[2]) + 25, w)
    y2 = min(int(area[3]) + 25, h)
    width = x2 - x1
    height = y2 - y1
    width_scale = w / width  # 假设原图像和剪切图像的宽度比例
    height_scale = h / height
    x_cropped = point[0]
    y_cropped = point[1]
    x_original = x_cropped * width_scale + x1
    y_original = y_cropped * height_scale + y1
    return [x_original, y_original]


def check_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1  # 第一个矩形框的左上角坐标 (x1, y1)，宽度 w1，高度 h1
    x2, y2, w2, h2 = rect2  # 第二个矩形框的左上角坐标 (x2, y2)，宽度 w2，高度 h2

    # 计算矩形框的右下角坐标
    r1_right = x1 + w1
    r1_bottom = y1 + h1
    r2_right = x2 + w2
    r2_bottom = y2 + h2

    # 判断两个矩形框是否有重叠
    if x1 < r2_right and r1_right > x2 and y1 < r2_bottom and r1_bottom > y2:
        return True  # 存在重叠
    else:
        return False  # 不存在重叠


def mask_generate(image, area):
    h, w = image.shape
    # 定义目标区域的坐标（左上角和右下角）
    x1 = max(int(area[0]) - 25, 0)
    y1 = max(int(area[1]) - 25, 0)
    x2 = min(int(area[2]) + 25, w)
    y2 = min(int(area[3]) + 25, h)

    # 从图像中剪切出目标区域
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image)
    mask[y1:y2, x1:x2] = 255
    mask = mask.astype(np.uint8)
    return mask


def mask_generate_temp(image, area):
    h, w = image.shape
    # 定义目标区域的坐标（左上角和右下角）
    x1 = max(int(area[0]) - 25, 0)
    y1 = max(int(area[1]) - 25, 0)
    x2 = min(int(area[2]) + 25, w)
    y2 = min(int(area[3]) + 25, h)

    # 从图像中剪切出目标区域
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image)
    mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    mask = mask.astype(np.uint8)
    return mask


def point_match(area, point):
    vertices = [[int(area[0]), int(area[1])], [int(area[2]), int(area[1])], [int(area[0]), int(area[3])],
                [int(area[2]), int(area[3])]]
    # 初始化最小距离和最近顶点索引
    min_distance = math.inf
    nearest_vertex = None

    # 遍历矩形的四个顶点
    for vertex in vertices:
        # 计算点与当前顶点之间的距离
        distance = math.sqrt((point[0] - vertex[0]) ** 2 + (point[1] - vertex[1]) ** 2)

        # 如果距离比当前最小距离小，则更新最小距离和最近顶点
        if distance < min_distance:
            min_distance = distance
            nearest_vertex = vertex
    if nearest_vertex == vertices[0]:
        return 1
    elif nearest_vertex == vertices[1]:
        return 2
    elif nearest_vertex == vertices[2]:
        return 3
    elif nearest_vertex == vertices[3]:
        return 4
    return 0


def is_approx_rectangle(points):
    # 计算所有边的长度
    side_lengths = []
    for i in range(len(points)):
        j = (i + 1) % len(points)
        side_lengths.append(distance(points[i], points[j]))

    # 计算对角线长度
    diagonal_lengths = []
    for i in range(len(points)):
        j = (i + 2) % len(points)
        diagonal_lengths.append(distance(points[i], points[j]))

    # 检查边长度的比值是否相似
    side_ratio = max(side_lengths) / min(side_lengths)
    diagonal_ratio = max(diagonal_lengths) / min(diagonal_lengths)
    ratio_threshold = 2.5  # 根据需要调整此阈值

    # 如果比值表明大致是一个矩形，则返回1，否则返回0
    if side_ratio <= ratio_threshold and diagonal_ratio <= ratio_threshold:
        return 1
    else:
        return 0


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
