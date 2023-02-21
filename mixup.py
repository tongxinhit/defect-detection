import cv2
import numpy as np
 
img1 = cv2.imread("/home/tongxin/dataaugument/for-mosica/pic/1.jpg")
img1 = cv2.resize(img1, (224, 224))
 
img2 = cv2.imread("/home/tongxin/dataaugument/for-mosica/pic/2.jpg")
img2 = cv2.resize(img2, (224, 224))
 
alpha = 1.0
lam = np.random.beta(alpha, alpha)
mixed_img = lam * img1 + (1 - lam) * img2
 
cv2.imwrite("/home/tongxin/dataaugument/for-mosica/mixup1.png", mixed_img)