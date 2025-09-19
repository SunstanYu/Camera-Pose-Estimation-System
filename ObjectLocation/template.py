import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('E:/Yolo/ultralytics/runs/detect/predict5/img5.jpg')
template = cv2.imread('template/bat.jpg')

th, tw = template.shape[:2]
rv = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(rv)

top_left = min_loc
bottom_right = (top_left[0] + tw, top_left[1] + th)
new_img = img.copy()
cv2.rectangle(new_img, top_left, bottom_right, 255, 2)

plt.subplot(131)
plt.imshow(template, cmap='gray')
plt.title('template')
plt.axis('off')

plt.subplot(132)
plt.imshow(rv, cmap='gray')
plt.title('matcing result')
plt.axis('off')

plt.subplot(133)
plt.imshow(new_img, cmap='gray')
plt.title('result')
plt.axis('off')

plt.show()
