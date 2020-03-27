import os, sys
import time
import re

import cv2
import numpy as np
from PIL import Image

from alpha_beta_filter import utils, DataAssociation, AlphaBetaFilter
from state import State

# Helper functions
def get_frame_id(fn):
    return int(re.sub(r"t(\d+)\.tif", "\\1", fn))

def get_average_video_frame(video_dir, SCALE_FACTOR, false_color=False):

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

def video_frame_iterator(video_dir, debug, SCALE_FACTOR):
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
                frame = cv2.imread("%s/%s" % (video_dir, fn), cv2.IMREAD_COLOR)
                new_shape = (frame.shape[1]//SCALE_FACTOR, frame.shape[0]//SCALE_FACTOR)
                frame = cv2.resize(frame, new_shape)
                yield (get_frame_id(fn), frame)

class CellDataLoader:
    def __init__(self, cell_data_dir, scale_factor=2, DEBUG=False):
        """
        TODO: image file path
              SCALE FACTOR
              and other hyper params?
        """
        self.localization = [] # TODO: create class to hold the position tuples?
        self.images = [] # this is wasting a lot of mem (TODO: traverse only when needed?)
        self.segmentation = [] # this is not used in further parts
        self.process(cell_data_dir, scale_factor, DEBUG)

    def process(self, cell_data_dir, scale_factor, DEBUG):

        avg_frame = get_average_video_frame(cell_data_dir, scale_factor, false_color=True)

        for frame_id, frame in video_frame_iterator(cell_data_dir, False, scale_factor):
            print(frame_id)

            frame_lum = np.array(Image.fromarray(frame).convert('L'))

            # Remove background bias
            frame_diff = cv2.absdiff(frame_lum, avg_frame)

            # Remove boundary noise; Adding mask
            roi_mask = np.ones((frame_diff.shape[0], frame_diff.shape[1]))
            roi_mask = roi_mask.astype(np.int8)
            roi_mask[0:15, 190:] = 0
            frame_roi = cv2.bitwise_and(frame_diff, frame_diff, mask=roi_mask)
            # cv2.imshow("mask", frame_roi)

            # Thresholding
            # frame_th = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 5)
            _, frame_th = cv2.threshold(frame_roi, 10, 255, cv2.THRESH_BINARY)
            frame_blur = cv2.GaussianBlur(frame_th, (3, 3), 0)
            # _, frame_th = cv2.threshold(frame_blur, 50, 255, cv2.THRESH_BINARY)
            

            frame_morph = cv2.morphologyEx(frame_th, cv2.MORPH_CLOSE, (7, 7), iterations=1)
            frame_morph = cv2.morphologyEx(frame_morph, cv2.MORPH_OPEN, (7, 7), iterations=1)
            frame_morph = cv2.morphologyEx(frame_morph, cv2.MORPH_CLOSE, (11, 11), iterations=1)

            frame_blur = cv2.GaussianBlur(frame_morph, (9, 9), 0)
            _, frame_blur = cv2.threshold(frame_blur, 50, 255, cv2.THRESH_BINARY)

            # # Flood filling
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_blur, 4, cv2.CV_32S)

            n_obj = 0
            cur_frame_locations = []
            for stat, cent in zip(stats[1:], centroids[1:]):
                x, y, w, h = stat[:4]
                if stat[4] < 100:
                    continue
                color = (0, 255, 0)
                n_obj += 1

                # Insert object centroid; TODO: data structure..
                state = State()
                state.set_centroid(x, y)
                state.set_bbox(w, h)
                cur_frame_locations.append(state)

                if DEBUG:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
                    cv2.putText(frame, "%.3f" % (stat[4]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=color) 

            if DEBUG:
                cv2.putText(frame, "Detected %i cells" % (n_obj), (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 0, 0)) 
                cv2.imshow("Lum_video", frame_blur)
                cv2.imshow("Orig_video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    exit(0)
                time.sleep(1./20)

            self.images.append(frame)
            self.localization.append(cur_frame_locations)

if __name__ == "__main__":
    data = CellDataLoader("../data/cell/CS585-Cells/", 2, DEBUG=False)
    print(data.localization[0][0].to_array())
    cell_tracker = AlphaBetaFilter(data, data_association_fn=DataAssociation.associate, window_size=(600,600), DEBUG=True)
    cell_tracker.run()