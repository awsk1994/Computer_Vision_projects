import os, sys
import time
import re

import cv2
import numpy as np
from PIL import Image

cv2.namedWindow("Orig_video")

# Constants
SCALE_FACTOR = 2
FPS = 5
DEBUG = False

def nothing(x):
    pass

def get_frame_id(fn):
    return int(re.sub(r".*?frame(\d+)\.ppm", "\\1", fn))

def get_next_new_color(usedColors):
    newColor = (np.random.choice(range(256), size=3))
    while np.any([np.all(uc == newColor) for uc in usedColors]): # if newColor matches any of the oldColors
        newColor = (np.random.choice(range(256), size=3))
    return newColor

def get_average_video_frame(video_dir, false_color=False):

    frames = []
    for _, _, file_list in os.walk(video_dir):
        file_list = sorted(file_list, key=lambda x: get_frame_id(x))

        for fn in file_list:
            frame = cv2.imread("%s/%s" % (video_dir, fn), cv2.IMREAD_COLOR)
            new_shape = (frame.shape[1]//SCALE_FACTOR, frame.shape[0]//SCALE_FACTOR)
            frame = cv2.resize(frame, new_shape)
            if false_color:
                frame = np.array(Image.fromarray(frame).convert('L'))
            frames.append(frame)
        print("shape", np.array(frames).shape)
        avg_frame = np.average(np.array(frames).astype(np.uint32), axis=0).astype(np.uint8)
        print(np.max(avg_frame), np.min(avg_frame))
        print("avg shape", avg_frame.shape)
        return avg_frame

def video_frame_iterator(video_dir, debug):
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

        print("obj: %i, area: %.3f, centroid: " % (i, stat[4]), c)
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

    avg_frame = get_average_video_frame("../CS585-BatImages/FalseColor/", false_color=True)
    # avg_false_color_frame = get_average_false_color_frame("./CS585-BatImages/FalseColor/")
    if DEBUG:
        cv2.imshow("AvgFrame", avg_frame)
        cv2.waitKey(0)
    for frame_id, frame in video_frame_iterator("../CS585-BatImages/FalseColor/", DEBUG):


        frame_lum = np.array(Image.fromarray(frame).convert('L'))
        
        # Remove background bias
        frame_diff = cv2.absdiff(frame_lum, avg_frame)

        # frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        
        # # Thresholding
        # _, frame_th = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        # frame_blur = cv2.GaussianBlur(frame_th, (3, 3), 0)
        # _, frame_th = cv2.threshold(frame_blur, 50, 255, cv2.THRESH_BINARY)
        
        # # Flood filling
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_th, 4, cv2.CV_32S)

        # for stat in stats[1:]:
        #     x, y, w, h = stat[:4]
        #     circularity = stat[4] / (w*h) #(((x**2+y**2)**0.5/2)**2*np.pi)
        #     if circularity < 0.5:
        #         color = (0, 255, 0)
        #     else:
        #         color = (0, 0, 255)
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
        #     cv2.putText(frame, "%.3f" % (circularity), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=color) 

        # cv2.imshow("diff_video", frame_th)
        cv2.imshow("Orig_video", frame)
        cv2.imshow("Lum_video", frame_diff)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            exit(0)
