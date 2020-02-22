'''
1. Add more template_scissor. Optimize detection - Alex
2. Skin Detection - Alex
3. Background subtraction - Shawn
4. multi-scale template_scissor matching - Shawn
5. Experiment template_scissor matching algorithm

'''
import cv2
import numpy as np

scale_percents = [0.4, 0.2, 0.1, 0.05]
frame_resize = 4
sel_r, sel_g, sel_b = None, None, None

cap = cv2.VideoCapture(0)
cv2.namedWindow("frame")

thres_val = 0.0001

def click_callback(event, x, y, flags, param):
	global sel_r, sel_g, sel_b

	if event == cv2.EVENT_LBUTTONUP:
		click_coord = (x,y)
		[sel_r, sel_g, sel_b] = orig_frame[y][x]
		print(click_coord)
		print(frame.shape)

def nothing(x):
	pass

cv2.setMouseCallback("frame", click_callback)
cv2.createTrackbar("thres_r", "frame",0,255, nothing) 	# 11
cv2.createTrackbar("thres_g", "frame",0,255, nothing)	# 190
cv2.createTrackbar("thres_b", "frame",0,255, nothing)	# 48

def get_smaller_frame(frame, frame_l1, scale_percent):
    dimen = (int(frame.shape[1] * scale_percent), int(frame.shape[0] * scale_percent))
    frame_img = cv2.resize(frame, dimen)

    print("frame_img", (frame_img.shape[0], frame_img.shape[1]), "entire_img shape", frame_l1.shape)

    entire_frame = np.zeros(frame_l1.shape, dtype='uint8')
    entire_frame[0:frame_img.shape[0], 0:frame_img.shape[1]] = frame_img
    return entire_frame

# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# methods2 = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

if __name__ == "__main__":
	template_scissor = cv2.imread('img/gesture_scissor.png', 0)
	template_scissor = cv2.resize(template_scissor, (template_scissor.shape[1]//10, template_scissor.shape[0]//10))
	cv2.imshow("template_sccisor", template_scissor)
	# template_scissor = cv2.filter2D(template_scissor,-1,kernel)

	template_point = cv2.imread('img/gesture_point.png', 0)
	template_point = cv2.resize(template_point, (template_point.shape[1]//10, template_point.shape[0]//10))
	cv2.imshow("template_point", template_point)

	while True:
		ret, frame = cap.read()
		frame = cv2.resize(frame, (frame.shape[1]//frame_resize, frame.shape[0]//frame_resize))
		orig_frame = frame.copy()

		# Skin Color
		thres_r=cv2.getTrackbarPos("thres_r", "frame")
		thres_g=cv2.getTrackbarPos("thres_g", "frame")
		thres_b=cv2.getTrackbarPos("thres_b", "frame")

		if sel_r is not None and sel_g is not None and sel_b is not None: 
			# Apply rgb threshold
			lower_range = np.array([sel_r - thres_r, sel_g - thres_g, sel_b - thres_b])
			upper_range = np.array([sel_r + thres_r, sel_g + thres_g, sel_b + thres_b])
			mask = cv2.inRange(frame, lower_range, upper_range);
			frame2 = mask
		else:
			frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame_gs = frame2 #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# frame_gs = cv2.resize(frame_gs, (frame_gs.shape[1]//frame_resize, frame_gs.shape[0]//frame_resize))
		cv2.imshow('frame', frame_gs)

		# Match Template
		res_point = cv2.matchTemplate(frame_gs, template_point, cv2.TM_CCOEFF_NORMED)
		res_point = np.abs(res_point)**3
		val, res_point = cv2.threshold(res_point, thres_val, 0, cv2.THRESH_TOZERO)
		res_point = cv2.normalize(res_point,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		max_loc_p = tuple(np.argwhere(res_point.max() == res_point)[0][::-1])
		# print('point', max_loc_p)
		# print(min_val, max_val, min_loc, max_loc)
		# print(res_point)
		res_point = cv2.resize(res_point, (res_point.shape[1], res_point.shape[0]))
		cv2.imshow("Point result", res_point)

		cv2.circle(orig_frame, max_loc_p, radius=20, color=(0, 255, 0), thickness=5)

		# scissor
		res_scissor = cv2.matchTemplate(frame_gs, template_scissor, cv2.TM_CCOEFF_NORMED)
		res_scissor = np.abs(res_scissor)**3
		val, res_scissor = cv2.threshold(res_scissor, thres_val, 0, cv2.THRESH_TOZERO)
		res_scissor = cv2.normalize(res_scissor,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		max_loc_p = tuple(np.argwhere(res_scissor.max() == res_scissor)[0][::-1])
		# print('point', max_loc_p)
		# print(min_val, max_val, min_loc, max_loc)
		# print(res_scissor)
		res_scissor = cv2.resize(res_scissor, (res_scissor.shape[1], res_scissor.shape[0]))
		cv2.imshow("Scissor result", res_scissor)

		cv2.circle(orig_frame, max_loc_p, radius=20, color=(255, 0, 0), thickness=5)
		
		cv2.imshow("color frame", orig_frame)
		
		# # Multi-dimension
		# dimen_l1 = (int(frame.shape[0] * scale_percents[0]), int(frame.shape[1] * scale_percents[0]))
		# frame_l1 = cv2.resize(frame, (dimen_l1[1], dimen_l1[0]))

		# mask = np.zeros(frame_l1.shape, dtype='uint8')

		# dimen_l2 = (int(frame.shape[0] * scale_percents[1]), int(frame.shape[1] * scale_percents[1]))
		# frame_l2_in = cv2.resize(frame, (dimen_l2[1],dimen_l2[0]))
		# mask[0:frame_l2_in.shape[0], 0:frame_l2_in.shape[1]] = frame_l2_in

		# dimen_l3 = (int(frame.shape[0] * scale_percents[2]), int(frame.shape[1] * scale_percents[2]))
		# frame_l3_in = cv2.resize(frame, (dimen_l3[1],dimen_l3[0]))
		# frame_l3_y_offset = frame_l2_in.shape[0] + frame_l3_in.shape[0]
		# mask[frame_l2_in.shape[0]:frame_l3_y_offset, 0:frame_l3_in.shape[1]] = frame_l3_in

		# dimen_l4 = (int(frame.shape[0] * scale_percents[3]), int(frame.shape[1] * scale_percents[3]))
		# frame_l4_in = cv2.resize(frame, (dimen_l4[1],dimen_l4[0]))
		# frame_14_offset = frame_l3_y_offset + frame_l4_in.shape[0]
		# mask[frame_l3_y_offset:frame_14_offset, 0:frame_l4_in.shape[1]] = frame_l4_in

		# frame_concatenated = np.concatenate((frame_l1, mask), axis=1)

		# cv2.imshow('frame', frame_concatenated)
		# # cv2.imshow("frame_ori", frame)
		# # cv2.imshow("frame_l4", frame_l4_in)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()