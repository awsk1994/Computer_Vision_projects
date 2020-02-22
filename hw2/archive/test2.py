import cv2
import numpy as np

sel_r, sel_g, sel_b = None, None, None
prev_frame = None
scale_percents = [0.4, 0.2, 0.1, 0.05]
orig_frame = None

def click_callback(event, x, y, flags, param):
	global sel_r, sel_g, sel_b

	if event == cv2.EVENT_LBUTTONUP:
		click_coord = (y,x)
		click_coord_with_offset = (y,x)
		print("click_coord_with_offset:", click_coord_with_offset, "frame", orig_frame.shape)
		[sel_r, sel_g, sel_b] = (orig_frame[click_coord_with_offset[0],click_coord_with_offset[1]])
		print("click | coord={}, color={}".format(click_coord, [sel_r, sel_g, sel_b]))

def nothing(x):
	pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_callback)
cv2.createTrackbar("thres_r", "frame",0,255, nothing) 	# 11
cv2.createTrackbar("thres_g", "frame",0,255, nothing)	# 190
cv2.createTrackbar("thres_b", "frame",0,255, nothing)	# 48
cv2.createTrackbar("toggle_motion_track", "frame",0,1, nothing)	# 48

def get_smaller_frame(frame, frame_l1, scale_percent):
    dimen = (int(frame.shape[1] * scale_percent), int(frame.shape[0] * scale_percent))
    frame_img = cv2.resize(frame, dimen)

    print("frame_img", (frame_img.shape[0], frame_img.shape[1]), "entire_img shape", frame_l1.shape)

    entire_frame = np.zeros(frame_l1.shape, dtype='uint8')
    entire_frame[0:frame_img.shape[0], 0:frame_img.shape[1]] = frame_img
    return entire_frame

while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, (int(frame.shape[1] * scale_percents[1]), int(frame.shape[0] * scale_percents[1])))
	orig_frame = frame.copy()

	# Skin Color Treshold
	thres_r=cv2.getTrackbarPos("thres_r", "frame")
	thres_g=cv2.getTrackbarPos("thres_g", "frame")
	thres_b=cv2.getTrackbarPos("thres_b", "frame")
	is_toggle_motion_track = cv2.getTrackbarPos("toggle_motion_track", "frame")

	if sel_r is not None and sel_g is not None and sel_b is not None: 
		# Apply rgb threshold
		lower_range = np.array([sel_r - thres_r, sel_g - thres_g, sel_b - thres_b])
		upper_range = np.array([sel_r + thres_r, sel_g + thres_g, sel_b + thres_b])
		mask = cv2.inRange(frame, lower_range, upper_range);
		frame_filtered = cv2.bitwise_and(frame, frame, mask= mask)

		if is_toggle_motion_track:
			diff = cv2.subtract(frame_filtered, prev_frame) #applies median blur before subtracting for noise reduction 
			frame = diff
		else:
			frame = frame_filtered

		prev_frame = frame_filtered

	# Multi-dimension
	frame_l1 = frame

	mask = np.zeros(frame_l1.shape, dtype='uint8')

	dimen_l2 = (int(frame.shape[0] * scale_percents[1]), int(frame.shape[1] * scale_percents[1]))
	frame_l2_in = cv2.resize(frame, (dimen_l2[1],dimen_l2[0]))
	mask[0:frame_l2_in.shape[0], 0:frame_l2_in.shape[1]] = frame_l2_in

	dimen_l3 = (int(frame.shape[0] * scale_percents[2]), int(frame.shape[1] * scale_percents[2]))
	frame_l3_in = cv2.resize(frame, (dimen_l3[1],dimen_l3[0]))
	frame_l3_y_offset = frame_l2_in.shape[0] + frame_l3_in.shape[0]
	mask[frame_l2_in.shape[0]:frame_l3_y_offset, 0:frame_l3_in.shape[1]] = frame_l3_in

	dimen_l4 = (int(frame.shape[0] * scale_percents[3]), int(frame.shape[1] * scale_percents[3]))
	frame_l4_in = cv2.resize(frame, (dimen_l4[1],dimen_l4[0]))
	frame_14_offset = frame_l3_y_offset + frame_l4_in.shape[0]
	mask[frame_l3_y_offset:frame_14_offset, 0:frame_l4_in.shape[1]] = frame_l4_in

	frame_concatenated = np.concatenate((frame_l1, mask), axis=1)

	cv2.imshow('frame', frame_concatenated)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()