import cv2
import numpy as np
def gamma_trans(img,gamma):#gamma大于1时图片变暗，小于1图片变亮
	#具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
	gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	#实现映射用的是Opencv的查表函数
	return cv2.LUT(img,gamma_table)
img = cv2.imread("/home/tongxin/dataaugument/for-mosica/pic/1.jpg", cv2.IMREAD_COLOR)    # 打开文件
 
# 执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
img_corrected = gamma_trans(img, 0.5)
cv2.imwrite("/home/tongxin/dataaugument/for-mosica/pic/rectification0.5.jpg",img_corrected)
img_corrected = gamma_trans(img, 2)
cv2.imwrite("/home/tongxin/dataaugument/for-mosica/pic/rectification2.jpg",img_corrected)
