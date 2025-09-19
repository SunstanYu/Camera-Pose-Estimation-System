import cv2
import numpy as np
import tkinter as tk


def sift_match_images(image1, image2):
    # 创建SIFT对象
    sift = cv2.xfeatures2d.SIFT_create()

    # 在图像中找到关键点和描述子
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher()

    # 使用KNN匹配算法，k设为2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 应用比值测试来筛选匹配项
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 绘制匹配项
    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    points_A = [keypoints1[m.queryIdx].pt for m in good_matches]
    points_B = [keypoints2[m.trainIdx].pt for m in good_matches]
    print(points_A)
    print(points_B)
    # matches = bf.match(descriptors1, descriptors2)
    #
    # matches = sorted(matches, key=lambda x: x.distance)
    #
    # match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], image2, flags=2)

    return match_img, points_A, points_B


# # 读取两幅图像
# image1 = cv2.imread('origin_img/measureddata/door numberIMG_2303(20230529-142703).JPG', 0)  # 灰度图像
# image2 = cv2.imread('template/card.jpg', 0)  # 灰度图像
#
# # 调用SIFT算法进行图像匹配
# result_img = sift_match_images(image1, image2)
# h, w, _ = result_img.shape
# root = tk.Tk()
# screen_width = root.winfo_screenwidth()  # 获取屏幕的宽度
# screen_height = root.winfo_screenheight()  # 获取屏幕的高度
#
# scale = min(screen_width / w, screen_height / h)  # 计算缩放比例
# img_resized = cv2.resize(result_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
# x = int((screen_width - img_resized.shape[1]) / 2)
# y = int((screen_height - img_resized.shape[0]) / 2)
# # 显示结果图像
# cv2.imshow('Image Matching Result', img_resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
