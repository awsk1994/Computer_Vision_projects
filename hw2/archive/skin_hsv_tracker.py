import cv2
import colorsys
import numpy as np

# Instantiate global variables
sel_hue, sel_sat, sel_value = None, None, None
prev_frame = None

# Methods
def click_callback(event, x, y, flags, param):
	global sel_hue, sel_sat, sel_value
	if event == cv2.EVENT_LBUTTONUP:
		click_coord = (x, y)
		click_pixel = frame[y][x]
		[sel_hue, sel_sat, sel_value] = colorsys.rgb_to_hsv(click_pixel[0], click_pixel[1], click_pixel[2])
		print("click | coord={}, color={}, hsv={}".format(click_coord, click_pixel, [sel_hue, sel_sat, sel_value]))

def nothing(x):
	pass

# Configure Video Frame
cap = cv2.VideoCapture(0)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_callback)
cv2.createTrackbar("max_hue", "frame",0,255, nothing) 	# 11
cv2.createTrackbar("max_sat", "frame",0,255, nothing)	# 190
cv2.createTrackbar("max_value", "frame",0,255, nothing)	# 48

# Main Execution
# Loop video frame until 'q' key is pressed
while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, (640, 480)) 

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	thres_upper_hue=cv2.getTrackbarPos("max_hue", "frame")
	thres_upper_sat=cv2.getTrackbarPos("max_sat", "frame")
	thres_upper_value=cv2.getTrackbarPos("max_value", "frame")

	if sel_hue is not None and sel_sat is not None and sel_value is not None: 
		# define range of blue color in HSV
		lower_range = np.array([sel_hue - thres_upper_hue, sel_sat - thres_upper_sat, sel_value - thres_upper_value])
		upper_range = np.array([sel_hue + thres_upper_hue, sel_sat + thres_upper_sat, sel_value + thres_upper_value])

		# Threshold the HSV image
		mask = cv2.inRange(hsv, lower_range, upper_range)
		mask = hsv

		if prev_frame is not None:
			print('show diff')
			diff = cv2.subtract(mask, prev_frame) #applies median blur before subtracting for noise reduction 
			cv2.imshow("frame", diff)
		else:
			print('show normal')
			cv2.imshow("frame", mask)
		prev_frame = mask

	else:
		cv2.imshow("frame", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to quit video capture
		break

# close all open windows
cv2.destroyAllWindows()
