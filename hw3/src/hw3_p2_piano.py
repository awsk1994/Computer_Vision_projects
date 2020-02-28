""" Code for HW3 part 2 pianist hand detection

Author: Shawn Lin (shawnlin@bu.edu)
        Alex Wong (awong1@bu.edu)
"""

import os, sys
import time
import re

import cv2
import numpy as np

cv2.namedWindow("Orig_video")

# Constants
SCALE_FACTOR = 4
FPS = 20
DEBUG = False

# Dummy callback function
def nothing(x):
    pass

# HSV filter bar
if DEBUG:
    cv2.createTrackbar("low_h", "Orig_video", 0, 255, nothing)   # 11
    cv2.createTrackbar("high_h", "Orig_video", 12, 255, nothing)    # 11
    cv2.createTrackbar("low_s", "Orig_video", 94, 255, nothing)  # 190
    cv2.createTrackbar("high_s", "Orig_video", 178, 255, nothing)    # 190
    cv2.createTrackbar("low_v", "Orig_video", 117, 255, nothing)  # 48
    cv2.createTrackbar("high_v", "Orig_video", 255, 255, nothing)    # 48

# Parse frame ID from file name 
def get_frame_id(fn):
    return int(re.sub(r"piano_(\d+)\.png", "\\1", fn))

# Get an unused new color
def get_next_new_color(usedColors):
    newColor = (np.random.choice(range(256), size=3))
    while np.any([np.all(uc == newColor) for uc in usedColors]): # if newColor matches any of the oldColors
        newColor = (np.random.choice(range(256), size=3))
    return newColor


def get_average_video_frame(video_dir):
    """ Compute the average video frame.
    
        video_dir -- name of the video directory
    """
    frames = []
    for _, _, file_list in os.walk(video_dir):
        file_list = sorted(file_list, key=lambda x: get_frame_id(x))

        for fn in file_list:
            frame = cv2.imread("%s/%s" % (video_dir, fn), cv2.IMREAD_COLOR)
            new_shape = (frame.shape[1]//SCALE_FACTOR, frame.shape[0]//SCALE_FACTOR)
            frame = cv2.resize(frame, new_shape)
            frames.append(frame)
        print("shape", np.array(frames).shape)
        avg_frame = np.average(np.array(frames).astype(np.uint32), axis=0).astype(np.uint8)
        print(np.max(avg_frame), np.min(avg_frame))
        print("avg shape", avg_frame.shape)
        return avg_frame

def video_frame_iterator(video_dir, debug):
    """ Parse and traverse all the files in the directory and return a generator
        object of frames.
    
    Arguments:
        video_dir -- directory name that contains all the frames
        debug -- debug flag (freeze on first frame if set to True)
    """
    for _, _, file_list in os.walk(video_dir):
        file_list = sorted(file_list, key=lambda x: get_frame_id(x))

        if debug:
            while True:
                frame = cv2.imread("%s/%s" % (video_dir, file_list[0]), cv2.IMREAD_COLOR)
                new_shape = (frame.shape[1]//SCALE_FACTOR, frame.shape[0]//SCALE_FACTOR)
                frame = cv2.resize(frame, new_shape)
                yield (0, frame)
        else:
            for fn in file_list:
                time.sleep(1./FPS)
                frame = cv2.imread("%s/%s" % (video_dir, fn), cv2.IMREAD_COLOR)
                new_shape = (frame.shape[1]//SCALE_FACTOR, frame.shape[0]//SCALE_FACTOR)
                frame = cv2.resize(frame, new_shape)
                yield (get_frame_id(fn), frame)

# 
def skin_masking(frame, debug):
    """ Perform HSV thresholding for skin detection
    
    Arguments:
        frame -- Input frame
        debug -- debug flag. Show tracking bar for tuning HSV threshold when
                 debug flag set to True.
    """
    if debug:
        low_h = cv2.getTrackbarPos("low_h", "Orig_video")
        high_h = cv2.getTrackbarPos("high_h", "Orig_video")
        low_s = cv2.getTrackbarPos("low_s", "Orig_video")
        high_s = cv2.getTrackbarPos("high_s", "Orig_video")
        low_v = cv2.getTrackbarPos("low_v", "Orig_video")
        high_v = cv2.getTrackbarPos("high_v", "Orig_video")
    else:
        low_h, high_h, low_s, high_s, low_v, high_v = [0, 12, 94, 178, 117, 255]

    lower_range = np.array([low_h, low_s, low_v], dtype= "uint8")
    upper_range = np.array([high_h, high_s, high_v], dtype ="uint8")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.inRange(frame, lower_range, upper_range);
    
    return frame

# Filter unrelated binary object and return label map
def get_label_map(frame_th, num_labels, labels, stats, centroids):
    label_map = np.zeros((frame_th.shape[0], frame_th.shape[1], 3), np.uint8)

    
    stats = [(i, stat) for i, stat in enumerate(stats)]

    # Remove small objects
    ignore_obj_idx = {i: True for i, stat in stats if stat[4] < 150.}
    ignore_obj_idx[0] = True # Ignore background object

    # Top-right object filter
    top_right_ctrd = [0, 480]
    top_right_idx = -1
    for c, (i, stat) in zip(centroids, stats):
        
        if i in ignore_obj_idx:
            continue

        # print("obj: %i, area: %.3f, centroid: " % (i, stat[4]), c)
        if c[0] >= top_right_ctrd[0] and c[1] <= top_right_ctrd[1]:
            top_right_ctrd = c
            top_right_idx = i

    # Ignore top-rightest object condition:
    cur_obj_cnt = len(stats) - len(ignore_obj_idx)
    if cur_obj_cnt >= 3:
        ignore_obj_idx[top_right_idx] = True
    elif 1 < cur_obj_cnt < 3 and max([s[4] for i, s in stats if i != 0]) > 400.:
        ignore_obj_idx[top_right_idx] = True

    # Only focus on coloring the top 2 largest object
    filtered_stats = list(filter(lambda x: x[0] not in ignore_obj_idx, stats))
    sorted_stats = sorted(filtered_stats, key=lambda x: x[1][4], reverse=True)[:2]

    print("sorted_filtered_stats:", sorted_stats)

    color_map = [np.array([0, 0, 0])]
    for _ in range(num_labels):
        color = get_next_new_color(color_map)
        color_map.append(color)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            color_label = labels[i][j]
            if color_label not in ignore_obj_idx:
                color = color_map[color_label]
                label_map[i][j] = color
    
    return label_map, sorted_stats


if __name__ == "__main__":

    avg_frame = get_average_video_frame("../CS585-PianoImages")
    if DEBUG:
        cv2.imshow("AvgFrame", avg_frame)

    cv2.waitKey(0)
    for frame_id, frame in video_frame_iterator("../CS585-PianoImages", DEBUG):
        
        # Moving region to ROI_mask
        frame_diff = cv2.absdiff(frame, avg_frame)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, roi_mask = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
        
        if DEBUG:
            cv2.imshow("mask", roi_mask)

        # Apply ROI mask on original frame
        frame_roi = cv2.bitwise_and(frame, frame, mask=roi_mask)

        # skin detection
        frame_gs = skin_masking(frame_roi, DEBUG)

        # Morphology
        frame_blur = cv2.GaussianBlur(frame_gs, (3, 3), 0)
        _, frame_th = cv2.threshold(frame_blur, 50, 255, cv2.THRESH_BINARY)
        frame_blur = cv2.GaussianBlur(frame_th, (5, 5), 0)
        _, frame_th = cv2.threshold(frame_blur, 80, 255, cv2.THRESH_BINARY)

        kernel_3x3 = np.ones((5, 5), np.uint8)
        kernel_7x7 = np.ones((3, 3), np.uint8)
        frame_gs = cv2.dilate(frame_gs, kernel_3x3, iterations=1)
        frame_gs = cv2.erode(frame_gs, kernel_3x3, iterations=1)
        frame_gs = cv2.dilate(frame_gs, kernel_7x7, iterations=1)
        frame_gs = cv2.erode(frame_gs, kernel_7x7, iterations=1)

        frame_gs = cv2.GaussianBlur(frame_gs, (3, 3), 0)
        _, frame_gs = cv2.threshold(frame_gs, 150, 255, cv2.THRESH_BINARY)

        # Flood filling with object stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_th, 4, cv2.CV_32S)

        # Object detection heuristics
        label_map, stats = get_label_map(frame_th, num_labels, labels, stats, centroids)
        
        # Draw hand bounding box
        for stat in stats:
            x, y, w, h = stat[1][:4]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Orig_video", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            exit(0)
