import numpy as np
import cv2
import tkinter as tk

def laplace_sharpening(img):
    # 定义Laplace算子
    laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # 进行卷积
    laplacian = cv2.filter2D(img, -1, laplacian_kernel)
    # 将图像与Laplace图像相加
    sharpened = cv2.addWeighted(img, 1.0, laplacian, -0.5, 0)
    return sharpened
#
# img = cv2.imread("origin_img/IMG_2213.JPG")
# img_out=laplace_sharpening(img)
# h, w, _ = img.shape
# root = tk.Tk()
# screen_width = root.winfo_screenwidth()  # 获取屏幕的宽度
# screen_height = root.winfo_screenheight()  # 获取屏幕的高度
# scale = min(screen_width / w, screen_height / h)  # 计算缩放比例
# img_resized = cv2.resize(img_out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
# x = int((screen_width - img_resized.shape[1]) / 2)
# y = int((screen_height - img_resized.shape[0]) / 2)
# cv2.imshow("output", img_resized)
# # cv2.resizeWindow('output', 800, 600)  # 设置宽度为800，高度为600
# cv2.waitKey(0)
# cv2.destroyAllWindows()