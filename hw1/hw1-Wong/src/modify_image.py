import numpy as np
import cv2
from matplotlib import pyplot as plt

def grayscale(a):
    return np.average(a, axis=2).astype(np.uint8)

def flip(a, debug=False):
	copy = np.zeros((a.shape[0], a.shape[1]))
	for x in range(a.shape[1]):
		copy[:,x] = a[:,a.shape[1]-x-1]
	return copy

def blur(a, kernel_size=7, debug=False):
	output = np.ones((a.shape[0], a.shape[1]))

	count = 0
	half_offset = int((kernel_size-1)/2)
	for y in range(a.shape[0]):
		for x in range(a.shape[1]):
			snippet = np.ones((kernel_size, kernel_size)) * 255
			for y_i in range(-1 * half_offset, half_offset+1):	# -3:3+1
				for x_i in range(-1 * half_offset, half_offset+1):
					if x >= half_offset and x <= a.shape[1]-half_offset-1 and y >= half_offset and y <= a.shape[0]-half_offset-1:
						snippet[y_i+half_offset][x_i+half_offset] = a[y+y_i][x+x_i]

			output[y][x] = np.mean(snippet).astype(int)

			if count % 10000 == 0 and debug:
				print("Blur Progress: ", count/(a.shape[0] * a.shape[1]))
			count += 1

	return output

img = cv2.imread('photo.jpg')

gray = grayscale(img)
cv2.imwrite("grayscale.jpg", gray)

flip = flip(gray)
cv2.imwrite("flip.jpg", flip)

blur_img = blur(gray, 7, debug=0)
cv2.imwrite("blur_img.jpg", blur_img)
