'''

TODO: 
 - centroid on drawing gesture
 - hsv detection by click
 - delete images in download_frame at the start of the program.

'''
import scipy
import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cv2.namedWindow("frame")

# Default Variable
frame_resize = 4

# Global Variables for draw_frame
draw_frame = None
detect_paper_count = 0
detect_download_file_count = 0
download_count = 0
refreshed_download = 0

# Click to get HSV value.
def click_callback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONUP:
		click_coord = (x,y)
		hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		print("hsv detected = {} (ignore h)".format(hsv_img[y][x] * 255))

def nothing(x):
	pass

cv2.setMouseCallback("frame", click_callback)

# HSV filter bar
cv2.createTrackbar("low_h", "frame", 0, 255, nothing) 	# 11
cv2.createTrackbar("high_h", "frame", 255, 255, nothing) 	# 11
cv2.createTrackbar("low_s", "frame", 51, 255, nothing)	# 190
cv2.createTrackbar("high_s", "frame", 255, 255, nothing)	# 190
cv2.createTrackbar("low_v", "frame", 79, 255, nothing)	# 48
cv2.createTrackbar("high_v", "frame", 255, 255, nothing)	# 48
cv2.createTrackbar("start_drawing", "frame", 0, 1, nothing)	# 48

def get_1d_boundary(proj, epsilon=10):
	x_tups = []
	trigger = False
	for i, x in enumerate(proj):
		if not trigger:
			if x < epsilon*255:
				continue
			if x >= epsilon*255:
				x_tups.append([i, None])
				trigger = True
		else:
			if x >= epsilon*255:
				continue
			if x < epsilon*255:
				x_tups[-1][-1] = i
				trigger = False
	
	return x_tups

def get_match_activation(cur_frame, template, method=cv2.TM_CCOEFF_NORMED):
	res = cv2.matchTemplate(cur_frame, template, method)
	res = np.abs(res)**3
	max_loc = tuple(np.argwhere(res.max() == res)[0][::-1])
	return res, max_loc

def skin_masking(frame):
	low_h = cv2.getTrackbarPos("low_h", "frame")
	high_h = cv2.getTrackbarPos("high_h", "frame")
	low_s = cv2.getTrackbarPos("low_s", "frame")
	high_s = cv2.getTrackbarPos("high_s", "frame")
	low_v = cv2.getTrackbarPos("low_v", "frame")
	high_v = cv2.getTrackbarPos("high_v", "frame")

	lower_range = np.array([low_h, low_s, low_v], dtype= "uint8")
	upper_range = np.array([high_h, high_s, high_v], dtype ="uint8")
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	frame = cv2.inRange(frame, lower_range, upper_range);
	
	return frame

def draw_on_frame(frame, x, y, template_name):
	global draw_frame, detect_paper_count, detect_download_file_count, refreshed_download, download_count
	if draw_frame is None:
		draw_frame = np.ones(frame.shape) * 255

	if template_name == "paper":
		refreshed_download = 1
		detect_paper_count += 1
		if detect_paper_count == 20:
			draw_frame = np.ones(frame.shape) * 255
			detect_paper_count = 0
		print("detect_paper_count={}".format(detect_paper_count))

	elif template_name == "point":
		refreshed_download = 1
		w = draw_frame.shape[1]
		x_axis_start = w-x
		x_axis_end = w-(x+5)
		if x_axis_end < 0:
			x_axis_end = 0
		draw_frame[y][x_axis_end:x_axis_start] = (0, 255, 0)
		draw_frame[y+1][x_axis_end:x_axis_start] = (0, 255, 0)
		draw_frame[y+2][x_axis_end:x_axis_start] = (0, 255, 0)
		detect_paper_count = 0

	elif template_name == "rock": 
		None	# Neural gesture
	elif template_name == "seven":
		# Download Frame
		if refreshed_download == 1:	# Need to have other gesture before download is available.
			detect_download_file_count += 1
			print("Detect_download_file_count = {}".format(detect_download_file_count))

			if detect_download_file_count == 20:	# Need to hold seven for 20 frames to start downloading.
				print("Download Frame_{}".format(download_count))
				cv2.imwrite("./download_frame/img_{}.png".format(download_count), draw_frame)
				download_count += 1

				refreshed_download = 0
				detect_download_file_count = 0
		else:
			print("Detected Seven Gesture | But cannot download. refreshed_download is 0")
	return draw_frame

if __name__ == "__main__":
	templates = []
	template_names = [
		"img/gesture_paper.png",
		"img/gesture_point.png",
		"img/gesture_rock.png",
		"img/gesture_seven.png"
	]
	template_alias = [
		"paper",
		"point",
		"rock",
		"seven"
	]

	for i, fn in enumerate(template_names):
		template = cv2.imread(fn, 0)
		template = cv2.resize(template, (50, int(template.shape[0]/template.shape[1]*50)))
		_, template = cv2.threshold(template, 1, 255, cv2.THRESH_BINARY)
		templates.append(template)

	while True:
		ret, f = cap.read()
		frame = cv2.resize(f, (f.shape[1]//frame_resize, f.shape[0]//frame_resize))
		frame_gs = skin_masking(frame)

		# Skin detection
		frame_blur = cv2.GaussianBlur(frame_gs, (9, 9), 0)
		_, frame_th = cv2.threshold(frame_blur, 50, 255, cv2.THRESH_BINARY)
		frame_blur = cv2.GaussianBlur(frame_th, (7, 7), 0)
		_, frame_th = cv2.threshold(frame_blur, 80, 255, cv2.THRESH_BINARY)

		# Morphology
		kernel_3x3 = np.ones((5, 5), np.uint8)
		kernel_7x7 = np.ones((3, 3), np.uint8)

		frame_gs = cv2.dilate(frame_gs, kernel_3x3, iterations=1)
		frame_gs = cv2.erode(frame_gs, kernel_3x3, iterations=1)
		frame_gs = cv2.dilate(frame_gs, kernel_7x7, iterations=1)
		frame_gs = cv2.erode(frame_gs, kernel_7x7, iterations=1)

		frame_gs = cv2.GaussianBlur(frame_gs, (7, 7), 0)
		_, frame_gs = cv2.threshold(frame_gs, 150, 255, cv2.THRESH_BINARY)	

		# Generate object proposals
		contours = cv2.findContours(frame_gs,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(contours)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

		mx = (0,0,0,0)      # biggest bounding box so far
		mx_area = 0
		for cont in cnts[:1]:
			x,y,w,h = cv2.boundingRect(cont)
			area = w*h
			if area > mx_area:
				mx = x,y,w,h
				mx_area = area
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
			# Template matching	
			max_match_score = -100
			max_match_idx = -1
			for i, (t, alias) in enumerate(zip(templates, template_alias)):
				t_resize = cv2.resize(t, (w, h))
				
				obj_region = frame_gs[y:y+h, x: x+w]

				res = cv2.matchTemplate(obj_region, t_resize, cv2.TM_CCOEFF_NORMED)
				# Flip the template horizontally and match again
				# TODO: turn the template to some angle and match again?
				res_flip = cv2.matchTemplate(obj_region, t_resize[:, ::-1], cv2.TM_CCOEFF_NORMED)

				match_score = max(res.max(), res_flip.max())
				if match_score > max_match_score:
					max_match_idx = i
					max_match_score = match_score

			cv2.putText(frame, "%s:%.3f" % (template_alias[max_match_idx], max_match_score), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(255, 0, 0)) 

		draw_frame = draw_on_frame(frame, x, y, template_alias[max_match_idx])

		# Update frame
		cv2.imshow('frame_gs', frame_gs)
		cv2.imshow('frame', frame)
		cv2.imshow('frame_draw', draw_frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()