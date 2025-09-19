import cv2
import tkinter as tk
import sharp
import numpy as np


def my_good_featuressToTrack(img_src, maxCorners, qualityLevel, minDistance, mask=None, blockSize=3,
                             k=0.04):
    corner_arr = np.zeros(img_src.shape)
    eigen_val = calc_corner_eigen_value(img_src, block_size=blockSize, aperture_size=3, k=k,
                                        borderType=cv2.BORDER_DEFAULT)
    max_v = np.max(eigen_val)
    thr_v = max(max_v * qualityLevel, 0)
    eigen_val[eigen_val < thr_v] = 0

    if mask is not None:
        x, y = np.where(mask == 0)
        eigen_val[x, y] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_val = cv2.dilate(eigen_val, kernel=kernel)
    x, y = np.where(eigen_val == dilate_val)
    index = np.where(eigen_val[x, y] != 0)
    x, y = x[index], y[index]

    corner_arr[x, y] = eigen_val[x, y]

    corner_arr = array_nms_circle(corner_arr, minDistance, maxCount=maxCorners)
    x, y = np.where(corner_arr != 0)
    out_arr = np.array(list(zip(y, x))).reshape(-1, 1, 2)

    return list(out_arr)


def calc_corner_eigen_value(img_src, block_size=2, aperture_size=3, k=0.04, borderType=cv2.BORDER_DEFAULT
                            ):
    if img_src.dtype != np.uint8:
        raise ("input image shoud be uint8 type")
    R_arr = np.zeros(img_src.shape, dtype=np.float32)
    img = img_src.astype(np.float32)
    scale = 1.0 / ((aperture_size - 1) * 2 * block_size * 255)
    Ix = cv2.Sobel(img, -1, dx=1, dy=0, ksize=aperture_size, scale=scale, borderType=borderType)
    Iy = cv2.Sobel(img, -1, dx=0, dy=1, ksize=aperture_size, scale=scale, borderType=borderType)
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    f_xx = cv2.boxFilter(Ixx, ddepth=-1, ksize=(block_size, block_size), anchor=(-1, -1), normalize=False,
                         borderType=borderType)
    f_yy = cv2.boxFilter(Iyy, ddepth=-1, ksize=(block_size, block_size), anchor=(-1, -1), normalize=False,
                         borderType=borderType)
    f_xy = cv2.boxFilter(Ixy, ddepth=-1, ksize=(block_size, block_size), anchor=(-1, -1), normalize=False,
                         borderType=borderType)
    radius = int((block_size - 1) / 2)
    N_pre = radius
    N_post = block_size - N_pre - 1
    row_s, col_s = N_pre, N_pre
    row_e, col_e = img.shape[0] - N_post, img.shape[1] - N_post

    for r in range(row_s, row_e):
        for c in range(col_s, col_e):
            sum_xx = f_xx[r, c]
            sum_yy = f_yy[r, c]
            sum_xy = f_xy[r, c]

            root_min = 0.5 * (sum_xx + sum_yy) - 0.5 * np.sqrt(
                (sum_xx - sum_yy) ** 2 + 4 * (sum_xy ** 2))
            R_arr[r, c] = root_min
    return R_arr


def array_nms_circle(dataIn, radius=1, maxCount=None):
    distance = np.zeros((radius * 2 + 1, radius * 2 + 1))
    x_label = np.arange(-radius, radius + 1)
    y_label = np.arange(-radius, radius + 1)
    for i in range(0, 2 * radius + 1):
        distance[i, :] = x_label[i] ** 2 + y_label ** 2
    x, y = np.where(distance <= radius ** 2)
    mask_x, mask_y = x - radius, y - radius
    data = dataIn.copy()
    rows, cols = data.shape
    out_arr = np.zeros(data.shape)
    count = 0
    while (np.max(data) > 0):
        r, c = np.unravel_index(data.argmax(), data.shape)
        zone_r, zone_c = r + mask_x, c + mask_y
        index1 = np.logical_and(zone_r >= 0, zone_r < rows)
        index2 = np.logical_and(zone_c >= 0, zone_c < cols)
        index = np.logical_and(index1, index2)
        zone_r, zone_c = zone_r[index], zone_c[index]
        out_arr[r, c] = data[r, c]
        data[zone_r, zone_c] = 0
        count = count + 1
        if (maxCount != None and count >= maxCount):
            break
    return out_arr
