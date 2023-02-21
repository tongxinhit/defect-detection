import cv2
import numpy as np
img = cv2.imread("/home/tongxin/dataaugument/for-mosica/pic/1.jpg")

# dst = cv2.GaussianBlur(img,ksize=(5,5),sigmaX=0,sigmaY=0)
# 创建毛玻璃特效
# 参数2：高斯核的宽和高（建议是奇数）
# 参数3：x和y轴的标准差
dst = cv2.GaussianBlur(img, (11, 11), 0)
cv2.imwrite("/home/tongxin/dataaugument/for-mosica/pic/gaussian_blur.jpg", dst)