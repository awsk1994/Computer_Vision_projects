""" Code for HW3 part 2 bat wing-spread detection

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
SCALE_FACTOR = 2
FPS = 20
DEBUG = False

# Dummy callback function
def nothing(x):
    pass

# Parse frame ID from file name 
def get_frame_id(fn):
    return int(re.sub(r".*?frame(\d+)\.ppm", "\\1", fn))

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

if __name__ == "__main__":

    avg_frame = get_average_video_frame("../CS585-BatImages/Gray/")
    if DEBUG:
        cv2.imshow("AvgFrame", avg_frame)
        cv2.waitKey(0)

    for frame_id, frame in video_frame_iterator("../CS585-BatImages/Gray/", DEBUG):

        # Remove background bias
        frame_diff = cv2.absdiff(frame, avg_frame)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        
        # Thresholding
        _, frame_th = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        frame_blur = cv2.GaussianBlur(frame_th, (3, 3), 0)
        _, frame_th = cv2.threshold(frame_blur, 50, 255, cv2.THRESH_BINARY)
        
        # Flood filling
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_th, 4, cv2.CV_32S)

        for stat in stats[1:]:
            x, y, w, h = stat[:4]
            circularity = stat[4] / (w*h)
            aspect_ratio = max(w, h) / min(w, h)
            spread = True

            # Filter out small objects
            if stat[4] < 10.:
                continue
            
            # Bat wing heuristics
            if circularity > 0.5: # if aspect_ratio > 2.0:
                color = (255, 0, 0)
                spread = False
            else:
                color = (0, 0, 255)

            # Annotate bat status
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
            cv2.putText(frame, "%.3f" % (aspect_ratio), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=color) 
            if spread:
                cv2.putText(frame, "spread", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 255, 0))
            else:
                cv2.putText(frame, "fold", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 255, 0)) 

        cv2.imshow("diff_video", frame_th)
        cv2.imshow("Orig_video", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            exit(0)
