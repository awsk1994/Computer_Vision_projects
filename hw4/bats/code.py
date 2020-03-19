'''
TODO: what is gating?
'''

import cv2 as cv2
import numpy as np
import glob
import random
import os

DEBUG = True


# Helper
def remove_from_array(lst, target):
    for idx in range(len(lst)):
        if target == lst[idx]:
            return lst[:idx] + lst[idx+1:]
    return lst

"""
Description : A class to handle bulk image loading
"""
class data_loader:
    """
    Description : Loads images and stores it in the class object

    Params
    ------
        image_path : string
            path of image resources
        localization_path : string
            path to localization resources
        gray : boolean
            Explicitly read in gray scale
    """
    def __init__(self,image_path,localization_path=None,segmentation_path=None,gray=False):
        self.images=[]  
        for _, _, file_list in os.walk(image_path):
            # for filename in glob.glob(image_path + '\*.ppm'):
            for name in file_list:
                filename = image_path + "/" + name
                if(gray):
                    img=cv2.imread(filename,0)
                else:
                    img=cv2.imread(filename)
                self.images.append(img)
        

        if(not(localization_path)):
            return
        self.localization = [] # 2d array - Num images x num detections in image
        # for filename in glob.glob(localization_path + '\*.txt'):
        for _, _, file_list in os.walk(localization_path):
            for name in file_list:
                filename = localization_path + "/" + name
                loc_data_tuples= []
                f = open(filename,'r')
                # self.localization.append(f.read().splitlines())
                # i want a list of tuples
                loc_data = f.read().splitlines()
                for det in loc_data:
                    # det is string
                    # strip ','
                    x,y = det.split(',')
                    # convert to tuple ints
                    # append to loc_data_tuples
                    # loc_data_tuples.append([int(y)-1,int(x)-1])
                    loc_data_tuples.append([int(y)-1,int(x)-1])
                self.localization.append(loc_data_tuples)


        if(not(segmentation_path)):
            return
        self.segmentation = [] # 3d array - Num images x image 
        # for filename in glob.glob(segmentation_path + '\*.txt'):
        for _, _, file_list in os.walk(segmentation_path):
            for name in file_list:
                filename = segmentation_path + "/" + name
                image = []
                f = open(filename,'r')
                string_image= f.read().splitlines()
                for row in string_image:
                    image.append([int(x) for x in row.split(',')])
                self.segmentation.append(image)


"""
Description : Class to perform bat tracking
"""
class bat_tracking:

    """
    Description : Initializer for class
    """
    def __init__(self):
        pass
    
    """
    Description : predict next position of object

    Params :
    --------
        x_prev : list
            positions of objects in prev frame
        v_prev: list
            velocities of objects in prev frame  
    Returns :
    --------
        x_pred : list
            predictions of positions of objects in current frame
    """
    def get_x_pred(self,x_prev,v_prev):
    # add only the measurements and not ids
    # assuming ids are sorted and do not change
        x_pred = []
        for i in range(len(x_prev)):
            x_pred.append([[x_prev[i][0][0]+v_prev[i][0][0],x_prev[i][0][1]+v_prev[i][0][1]],x_prev[i][1]])
        return x_pred

    """
    Description : calculates euclidean distance between two points

    Params :
    --------
        a : list
            x,y coords of object a
        b: list
            x,y coords of object b   
    Returns :
    --------
        dist : float
            euclidian distance between 'a' and 'b'
    """
    def distance(self,a,b):
        import math
        # euclidien listance
        return np.linalg.norm(np.array(a)-np.array(b))
        # x_diff = a[0] - b[0]
        # y_diff = a[1] - b[1]
        # return math.pow(math.pow(x_diff, 2) + math.pow(y_diff, 2), 0.5)

    """
    Description : Performs the data association between new measurements and existing objects

    Params :
    --------
        x_pred : list
            predictions of object movements
        frame_measurements : list
            measurements of object movements
        v_pred : list
            predictions of object velocities
        num_objects : int
            counter to keep track of number of objects gloablly
    
    Returns :
    --------
        cur_measurements : list
            current measurements associated with objects
    """
    def association(self, x_pred, frame_measurements, v_pred):
        # to_delete = []
        # measurement_taken = np.zeros((len(frame_measurements)))

        # for i in range(len(x_pred)):                        # TODO: swap i and j?
        #     minn = float('inf')
        #     assigned = None
        #     for j, meas in enumerate(frame_measurements):
        #         # if(measurement_taken[j]):                   # TODO: potential problem
        #         #     continue
        #         dist = self.distance(x_pred[i][0],meas)     # distance from x_pred (prediction) to measurement
        #         # if(dist<=self.gating and dist<minn):
        #         if dist < minn:
        #             minn = dist
        #             assigned = j
        #     if(assigned is not None):                        # TODO: if assigned
        #         measurement_taken[assigned]=1
        #         cur_frame_x_pred_labels.append([frame_measurements[assigned],x_pred[i][1]])
        #     else:
        #         to_delete.append(i)

        print("association | len(x_pred)={}, len(frame_measurements)={}, should expect len(cur_frame_x_pred_labels + new_locs)={}".format(len(x_pred), len(frame_measurements), max(len(x_pred), len(frame_measurements))))
 
        cur_frame_x_pred_labels = []


        # For each localization point, compute the closest x_pred point. Assign to hash.
        x_pred_locs_hash = {} # {'i_key': [(j_key, dist),...]}, i_key = x_pred_key, j_keys = localization point
        for j, meas in enumerate(frame_measurements):       # Start with localization point
            min_dist = float('inf')
            closest_x_pred_label = None

            # Get closets x_pred
            for _, x_pred_data in enumerate(x_pred):
                x_pred_coord, x_pred_label = x_pred_data

                dist = self.distance(x_pred_coord, meas)     # distance from x_pred (prediction) to localization point
                if dist < min_dist:
                    if x_pred_label == 61:
                        print("c1 | x_pred_coord={}, loc_coord={}, dist={}".format(x_pred_coord, meas, dist))

                    min_dist = dist
                    closest_x_pred_label = x_pred_label

            # Add to x_pred_locs_hash
            if closest_x_pred_label in x_pred_locs_hash.keys():
                x_pred_locs_hash[closest_x_pred_label].append((j, min_dist, meas))
            else:
                x_pred_locs_hash[closest_x_pred_label] = [(j, min_dist, meas)]

        print("x_pred_locs_hash:", x_pred_locs_hash)

        # Shorten Hash so each each x_pred only has 1 locs. The rest are zombies. 
        zombie_locs = []
        for x_pred_key in x_pred_locs_hash.keys():
            loc_datas = x_pred_locs_hash[x_pred_key]
            if len(loc_datas) == 1:         # this x_pred only has 1 loc. Add to cur_frame_x_pred_labels
                loc_idx = loc_datas[0][0]
                # print("loc_idx={}".format(loc_idx))
                loc_coord = frame_measurements[loc_idx]
                cur_frame_x_pred_labels.append([loc_coord, x_pred_key])
            else:
                loc_idxs = [loc_data[0] for loc_data in loc_datas]
                min_idx = None
                min_dist = float('inf')
                for loc_data in loc_datas:
                    loc_idx, loc_dist, _ = loc_data
                    if loc_dist < min_dist:
                        min_dist = loc_dist
                        min_idx = loc_idx

                x_pred_locs_hash[x_pred_key] = [(min_idx, min_dist, None)]        # Update x_pred_locs_hash (not needed actually)
                loc_coord = frame_measurements[min_idx]
                cur_frame_x_pred_labels.append([loc_coord, x_pred_key])     # Update cur_frame_x_pred_labels

                zombie_locs += remove_from_array(loc_idxs, min_idx) # Zombie locs

        print("cur_frame_x_pred_labels: ", cur_frame_x_pred_labels)

        # Zombie Locs are the new objects
        new_locs = []
        x_pred_labels_np = np.array([x_pred_data[1] for x_pred_data in x_pred])
        new_x_pred_label = np.max(x_pred_labels_np) + 1

        print("c5 | x_pred_labels_np={}, new_x_pred_label={}".format(x_pred_labels_np, new_x_pred_label))

        for zombie_idx in range(len(zombie_locs)):
            loc_idx = zombie_locs[zombie_idx]
            loc_coord = frame_measurements[loc_idx]
            new_x_pred_label = new_x_pred_label

            # x_pred.append([loc_coord, new_x_pred_label])
            # v_pred.append([[0,0], new_x_pred_label])                        # set zero velocity for new objects
            # cur_frame_x_pred_labels.append([loc_coord, new_x_pred_label])   # Update cur_frame_x_pred_labels
            new_locs.append([loc_coord, new_x_pred_label])
            new_x_pred_label += 1

        print("c4 | new_locs: ", new_locs)
        print("association | len(cur_frame_x_pred_labels)={}, len(new_locs)={}".format(len(cur_frame_x_pred_labels), len(new_locs)))
        return cur_frame_x_pred_labels, new_locs

        #     if(measurement_taken[i]==0):
        #         x_pred.append([[0,0],num_objects+1]) # to account for new entries, kind of like prior
        #         v_pred.append([[0,0],num_objects+1]) # to account for new entries, kind of like prior
        #         cur_frame_x_pred_labels.append([frame_measurements[i],num_objects+1])
        #         num_objects = num_objects + 1
        
        # # some x_pred may dissapear
        # # delete elements indexed in to_delete
        # for i in sorted(to_delete, reverse=True):
        #     del x_pred[i]
        #     del v_pred[i]
        # return cur_frame_x_pred_labels

    """
    Description : Performs the subtraction between predictions and measurements

    Params :
    --------
        frame_measurements : list
            measurements of object movements
        x_pred : list
            predictions of object movements
    
    Returns :
    --------
        res : list
            difference between measurements and predictions
    """
    def subtract(self, frame_measurements, x_preds):
        # perform object wise subtraction
        res = []
        # the below assumes element wise alignment, this may or may not hold
        for _, x_pred in enumerate(x_preds):
            x_pred_coord, x_pred_label = x_pred

            for _, frame_measurement in enumerate(frame_measurements):
                fm_coord, fm_label = frame_measurement

                if x_pred_label == fm_label:                    # TODO: Optimize
                    x_residual = fm_coord[0] - x_pred_coord[0]
                    y_residual = fm_coord[1] - x_pred_coord[1]
                    residual = (x_residual, y_residual)
                    res.append([residual, x_pred_label])
                    if x_pred_label == 61:
                        print("x_pred_label={}, fm_label={}, x_residual={}, y_residual={}".format(x_pred_label, fm_label, x_residual, y_residual))

        return res

    """
    Description : Performs the update step of alpha-beta filter

    Params :
    --------
        pred : list
            predictions of object movements/velocities
        alpha : float
            a parameter of the alpha beta filter (could either be alpha/beta)
        res : list
            difference between measurements and predictions
    
    Returns :
    --------
        est : list
            estimated positions/velocities
    """
    def update_coord(self, x_preds, alpha, residuals, new_locs):
        residual_hash = {}
        for _, residual in enumerate(residuals):
            residual_coord, residual_label = residual
            residual_hash[residual_label] = residual_coord

        # perform object wise subtraction
        results = []
        # the below assumes element wise alignment, this may or may not hold
        for _, x_pred in enumerate(x_preds):
            x_pred_coord, x_pred_label = x_pred

            if x_pred_label in residual_hash.keys():
                residual = residual_hash[x_pred_label]
                est_x_coord, est_y_coord = x_pred_coord[0] + alpha * residual[0], x_pred_coord[1] + alpha * residual[1]
                est_coord = (est_x_coord, est_y_coord)
                results.append([est_coord, x_pred_label])
                if x_pred_label == 61:
                    print("C2 | x_pred_label={}, residual={}, x_pred_coord={}, est_coord={}".format(x_pred_label, residual, x_pred_coord, est_coord))

            # # print("x_pred_coord", x_pred_coord)
            # for _, residual in enumerate(residuals):
            #     residual, residual_label = residual

            #     if x_pred_label == residual_label:
            #         est_x_coord, est_y_coord = x_pred_coord[0] + alpha * residual[0], x_pred_coord[1] + alpha * residual[1]
            #         est_coord = (est_x_coord, est_y_coord)
            #         results.append([est_coord, x_pred_label])
            #         break

        for _, new_loc in enumerate(new_locs):
            loc_coord, loc_label = new_loc
            if loc_label == 61:
                print("c4 | loc_label={}, loc_coord={}".format(loc_label, loc_coord))

            results.append([loc_coord, loc_label])  # default 0 velocity.

        print("c3 | x_est={}".format(results))
        return results

    """
    Description : Performs the update step of alpha-beta filter

    Params :
    --------
        pred : list
            predictions of object movements/velocities
        alpha : float
            a parameter of the alpha beta filter (could either be alpha/beta)
        res : list
            difference between measurements and predictions
    
    Returns :
    --------
        est : list
            estimated positions/velocities
    """
    def update_velocity(self, v_preds, beta, residuals, new_locs):
        residual_hash = {}
        for _, residual in enumerate(residuals):
            residual_coord, residual_label = residual
            residual_hash[residual_label] = residual_coord

        # perform object wise subtraction
        results = []
        # the below assumes element wise alignment, this may or may not hold
        for _, v_pred in enumerate(v_preds):
            v_pred_coord, v_pred_label = v_pred

            if v_pred_label in residual_hash.keys():
                residual = residual_hash[v_pred_label]
                est_x_coord, est_y_coord = v_pred_coord[0] + beta * residual[0], v_pred_coord[1] + beta * residual[1]
                est_coord = (est_x_coord, est_y_coord)
                results.append([est_coord, v_pred_label])

        for _, new_loc in enumerate(new_locs):
            loc_coord, loc_label = new_loc
            results.append([[0,0], loc_label])  # default 0 velocity.

        return results

    """
    Description : Function to draw tracks on the bat dataset

    Params :
    --------
        x_pos_comiled : list
            contains all trackings seen so far
        color_hash : dict
            dict of keys and random colors
        frame : numpy array
            current frame
    
    Returns :
    --------
        frame : numpy array
            frame with tracks drawn on it
    """
    def draw(self,x_pos_compiled,color_hash,frame):
        hash= {}
        for frame_num in range(len(x_pos_compiled)):
            for detection in x_pos_compiled[frame_num]:
                if detection[1] in hash:
                    hash[detection[1]]['y'].append(detection[0][0])
                    hash[detection[1]]['x'].append(detection[0][1])
                else:
                    hash[detection[1]]={'y':[detection[0][0]],'x':[detection[0][1]]}

        # iter = 0
        for key in hash:
            y= hash[key]['y']
            x= hash[key]['x']
            # iter = iter + 1
            # print('key is',key)
            for i in range(len(x)):
                color = color_hash[key]
                frame = cv2.circle(frame, (x[i],y[i]), 5, color, -1)
            # if iter>50:
            #     break
        return frame

    def draw_line(self, x_from, x_to, color_hash, frame):

        print("len(x_from)={}, len(x_to)={}".format(len(x_from), len(x_to)))

        x_from_hash = {}
        for _, x_from_data in enumerate(x_from):
            x_from_coord, x_from_label = x_from_data
            if x_from_label in x_from_hash.keys():
                x_from_hash[x_from_label]['y'] = (x_from_coord[0])
                x_from_hash[x_from_label]['x'] = (x_from_coord[1])
            else:
                x_from_hash[x_from_label]={'y':x_from_coord[0],'x':x_from_coord[1]}

        x_to_hash = {}
        for _, x_to_data in enumerate(x_to):
            x_to_coord, x_to_label = x_to_data
            if x_to_label in x_to_hash.keys():
                x_to_hash[x_to_label]['y'] = x_to_coord[0]
                x_to_hash[x_to_label]['x'] = x_to_coord[1]
            else:
                x_to_hash[x_to_label]={'y':x_to_coord[0],'x':x_to_coord[1]}

        # iter = 0
        max_diff_x, max_diff_y = 0, 0
        for key in x_to_hash.keys():
            x_to_coord = (x_to_hash[key]['x'], x_to_hash[key]['y'])
            x_from_coord = (x_from_hash[key]['x'], x_from_hash[key]['y']) if key in x_from_hash.keys() else x_to_coord

            color = color_hash[key]

            # rect_start_pt = (x_prev_hash_x[i], x_prev_hash_y[i])
            # rect_end_pt = (x_prev_hash_x[i]+1, x_prev_hash_y[i]+1)
            # frame = cv2.rectangle(frame, start_pt, end_pt, color, 4)  
            # frame = cv2.circle(frame, (x_prev_hash_x[i],x_prev_hash_y[i]), 4, color, -1)
            frame = cv2.circle(frame, x_to_coord, 4, color, -1)
            frame = cv2.line(frame, x_to_coord, x_from_coord, color, 2)
            frame = cv2.putText(frame, str(key), x_from_coord, cv2.FONT_HERSHEY_COMPLEX, 1, color)

            diff_x, diff_y = abs(x_to_coord[0] - x_from_coord[0]), abs(x_to_coord[1] - x_from_coord[1])
            max_diff_x, max_diff_y = max(max_diff_x, diff_x), max(max_diff_y, diff_y)

        print("max_diff_x={}, max_diff_y={}".format(max_diff_x, max_diff_y))
            # if iter>50:
            #     break
        return frame

    def create_color_hash(self, upper_range=1000):
        # print('creating color hash')
        color_hash ={}
        for k in range(upper_range):
            color_hash[k]=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        # print('color hash ready')
        return color_hash
    """
    Description : Function to show bat images with centroid data
    """
    def show_frame_wise(self):
        data = data_loader(image_path='./CS585-BatImages/Gray', localization_path='./Localization', segmentation_path=None) #segmentation_path='./Segmentation'
        print('Finished loading data')
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.namedWindow('orig image',cv2.WINDOW_NORMAL)

        # Default Initialization
        alpha = beta = 1
        x_prev, v_prev, x_pos_compiled = [], [], []
        self.gating = float('inf') # no gating for first run, how will you define gating, assume gate = 5 px circle
        color_hash = self.create_color_hash()
        num_objects = len(data.localization[0])

        # Initialize x_prev, v_prev
        for i in range(len(data.localization[0])):
            x,y = data.localization[0][i]
            x_prev.append([[x,y],i])
            v_prev.append([[0,0],i])

        # Process each frame
        for i,frame in enumerate(data.images):
            orig_frame = frame.copy()

            x_orig_prev = x_prev.copy()

            # Step 1: Predict
            x_pred = self.get_x_pred(x_prev, v_prev)
            v_pred = v_prev
            if DEBUG:
                print('frame', i, '| Step 1 | x_pred after initial prediction',x_pred)
                print('frame', i, '| Step 1 | v_pred after initial prediction',v_pred)

            # Step 2: Associate object across prev frame and current frame.
            if DEBUG:
                print('frame', i, "| step 2 | Localization: ", data.localization[i])
            cur_measurements, new_locs = self.association(x_pred, data.localization[i], v_pred)
            if DEBUG:
                print('frame', i, "| step 2 | cur_measurements: ",cur_measurements)
                print('frame', i, "| step 2 | x_pred: ",x_pred)

            # Step 3: Prediction Error Adjustment (Residual)
            # the above function also changes x_pred, v_pred,num_objects
            res = self.subtract(cur_measurements, x_pred)
            print("done substract")

            x_est = self.update_coord(x_pred, alpha, res, new_locs)
            print("done update coord")

            if(i!=0):  # neccesary because other wise velocity is overestimated in first frame
                v_est = self.update_velocity(v_pred, beta, res, new_locs)
            else:
                v_est = v_pred
            print("done update velocity")

            if DEBUG:
                print('frame', i, ' | step 3 | res',res)
                print('frame', i, ' | step 3 | x_est',x_est)
                print('frame', i, ' | step 3 | v_est',v_est)

            v_prev = v_est
            x_prev = x_est
            x_pos_compiled.append(x_est)

            if DEBUG:
                print('Step 4 | Finished tracking')
                print('Step 4 | Num objects so far',num_objects)

                print("Step 5 | Drawing")
            # frame = self.draw(x_pos_compiled,color_hash,frame)
            this_frame = frame.copy()
            this_frame = self.draw_line(x_orig_prev, x_est, color_hash, this_frame)

            orig_frame = self.draw(x_pos_compiled, color_hash, orig_frame)

            while (True):
                cv2.imshow("image", this_frame)
                cv2.resizeWindow('image',600,600)
                cv2.imshow("orig image",orig_frame)
                cv2.resizeWindow('orig image',600,600)
                if cv2.waitKey(1) & 0xFF == ord('q'):   # Press q to go to next frame
                    break
            self.gating = 50
        cv2.destroyAllWindows()

bat_track = bat_tracking()
bat_track.show_frame_wise()

# to do - solve the confusing tracks thing
# remove tracks that are dead
# read Yifu's thing fully