import cv2
import numpy as np
from matplotlib import pyplot as plt


def blur(img, kernel_size=6):
	kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size * kernel_size)
	img_blur = cv2.filter2D(img,-1,kernel)
	return img_blur

img_rgb = cv2.imread('img/selfie1.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# img_gray = blur(img_gray)
# img_gray = cv2.resize(img_gray, (img_gray.shape[1]//2,img_gray.shape[0]//2))
template = cv2.imread('img/selfie1_scissor_hand.png',0)
# template = cv2.resize(template, (template.shape[1]//2,template.shape[0]//2))

plt.imshow(img_gray)
plt.show()
plt.imshow(template)
plt.show()

w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
print(res)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

plt.imshow(img_rgb)
plt.show()

