import cv2 as cv2
import numpy as np
import glob
import random

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
        for filename in glob.glob(image_path + '\*.ppm'):
            if(gray):
                img=cv2.imread(filename,0)
            else:
                img=cv2.imread(filename)
            self.images.append(img)
        
        if(not(localization_path)):
            return
        self.localization = [] # 2d array - Num images x num detections in image
        for filename in glob.glob(localization_path + '\*.txt'):
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
        for filename in glob.glob(segmentation_path + '\*.txt'):
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
        # euclidien listance
        return np.linalg.norm(np.array(a)-np.array(b))

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
    def association(self,x_pred, frame_measurements,v_pred,num_objects):
        to_delete = []
        cur_measurements = []
        measurement_taken=np.zeros((len(frame_measurements)))
        for i in range(len(x_pred)):
            minn = float('inf')
            assigned = None
            for j,meas in enumerate(frame_measurements):
                if(measurement_taken[j]):
                    continue
                dist = self.distance(x_pred[i][0],meas)
                if(dist<=self.gating and dist<minn):
                    minn = dist
                    assigned = j
            if(not(assigned)==None):
                measurement_taken[assigned]=1
                cur_measurements.append([frame_measurements[assigned],x_pred[i][1]])
            else:
                to_delete.append(i)
                
        # some measurements may not be taken - new objects
        for i in range(len(measurement_taken)):
            if(measurement_taken[i]==0):
                x_pred.append([[0,0],num_objects+1]) # to account for new entries, kind of like prior
                v_pred.append([[0,0],num_objects+1]) # to account for new entries, kind of like prior
                cur_measurements.append([frame_measurements[i],num_objects+1])
                num_objects = num_objects + 1
        
        # some x_pred may dissapear
        # delete elements indexed in to_delete
        for i in sorted(to_delete, reverse=True):
            del x_pred[i]
            del v_pred[i]
        return cur_measurements

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
    def subtract(self,frame_measurements,x_pred):
        # perform object wise subtraction
        res = []
        # the below assumes element wise alignment, this may or may not hold
        for i in range(len(x_pred)):
            res.append([[frame_measurements[i][0][0]-x_pred[i][0][0],frame_measurements[i][0][1]-x_pred[i][0][1]],x_pred[i][1]])
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
    def update(self,pred,alpha,res):
        est = []
        for i in range(len(pred)):
            est.append([[pred[i][0][0] + alpha * res[i][0][0],pred[i][0][1] + alpha * res[i][0][1]],pred[i][1]])
        return est
    
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

    """
    Description : Function to show bat images with centroid data
    """
    def show_frame_wise(self):
        data=data_loader(image_path='./CS585-BatImages/Gray',localization_path='./Localization',segmentation_path=None) #segmentation_path='./Segmentation'
        print('Finished loading data')
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.namedWindow('orig image',cv2.WINDOW_NORMAL)
        alpha  = beta = 1
        x_prev = []
        v_prev = []
        num_objects = len(data.localization[0])
        for i in range(len(data.localization[0])):
            x_prev.append([[0,0],i+1])
            v_prev.append([[0,0],i+1])
        # how will you define gating ?
        # assume gate = 5 px circle
        self.gating = float('inf') # no gating for first run
        x_pos_compiled = []
        print('creating color hash')
        color_hash ={}
        for k in range(300):
            color_hash[k]=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        print('color hash ready')
        for i,frame in enumerate(data.images):
            orig_frame = frame.copy()
            x_pred = self.get_x_pred(x_prev, v_prev)
            print('x_pred after initial prediction',x_pred)
            v_pred = v_prev
            print('v_pred after initial prediction',v_pred)
            cur_measurements = self.association(x_pred,data.localization[i],v_pred,num_objects)
            print('cur_measurements',cur_measurements)
            # the above function also changes x_pred, v_pred,num_objects
            res = self.subtract(cur_measurements,x_pred)
            print('res',res)
            x_est = self.update(x_pred,alpha,res)
            print('x_est',x_est)
            if(i!=0):  # neccesary because other wise velocity is overestimated in first frame
                v_est = self.update(v_pred,beta,res)
            else:
                v_est = v_pred
            print('v_est',v_est)
            v_prev = v_est
            x_prev = x_est
            x_pos_compiled.append(x_est)
            print('Finished tracking')
            print('Num objects so far',num_objects)
            print('Drawing started')
            frame= self.draw(x_pos_compiled,color_hash,frame)
            print('Drawing done')
            while (True):
                cv2.imshow("image",frame)
                cv2.resizeWindow('image',600,600)
                cv2.imshow("orig image",orig_frame)
                cv2.resizeWindow('orig image',600,600)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.gating = 50
        cv2.destroyAllWindows()

bat_track = bat_tracking()
bat_track.show_frame_wise()

# to do - solve the confusing tracks thing
# remove tracks that are dead
# read Yifu's thing fully