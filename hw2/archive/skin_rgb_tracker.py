import cv2
import colorsys
import numpy as np

sel_r, sel_g, sel_b = None, None, None
prev_frame = None
scale_percent = 0.5

def click_callback(event, x, y, flags, param):
	global sel_r, sel_g, sel_b

	if event == cv2.EVENT_LBUTTONUP:
		click_coord = (x, y)
		[sel_r, sel_g, sel_b] = (frame[y][x])
		print("click | coord={}, color={}".format(click_coord, [sel_r, sel_g, sel_b]))

def nothing(x):
	pass

# Configure Video Frame
cap = cv2.VideoCapture(0)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_callback)
cv2.createTrackbar("thres_r", "frame",0,255, nothing) 	# 11
cv2.createTrackbar("thres_g", "frame",0,255, nothing)	# 190
cv2.createTrackbar("thres_b", "frame",0,255, nothing)	# 48
cv2.createTrackbar("toggle_motion_track", "frame",0,1, nothing)	# 48

# keep looping until the 'q' key is pressed
while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, (int(frame.shape[1] * scale_percent), int(frame.shape[0] * scale_percent))) 

	thres_r=cv2.getTrackbarPos("thres_r", "frame")
	thres_g=cv2.getTrackbarPos("thres_g", "frame")
	thres_b=cv2.getTrackbarPos("thres_b", "frame")
	is_toggle_motion_track = cv2.getTrackbarPos("toggle_motion_track", "frame")

	if sel_r is not None and sel_g is not None and sel_b is not None: 
		# Apply rgb threshold
		lower_range = np.array([sel_r - thres_r, sel_g - thres_g, sel_b - thres_b])
		upper_range = np.array([sel_r + thres_r, sel_g + thres_g, sel_b + thres_b])
		mask = cv2.inRange(frame, lower_range, upper_range);
		frame_filtered = mask #cv2.bitwise_and(frame, frame, mask= mask)

		if is_toggle_motion_track:
			diff = cv2.subtract(frame_filtered, prev_frame) #applies median blur before subtracting for noise reduction 
			cv2.imshow("frame", diff)
		else:
			cv2.imshow("frame", frame_filtered)

		prev_frame = frame_filtered
	else:
		cv2.imshow("frame", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
