import numpy as np
import cv2

from cell_data_loader import CellDataLoader
from bat_data_loader import BatDataLoader

from data_association import GNNSF

np.set_printoptions(precision=3, suppress=True)

# Helper func
def get_coord(x):
    return (int(x[0]), int(x[1]))

class KalmanFilter:
    def __init__(self, P=10, G_thresh=120, std_dev=1, random_seed=30):
        """
        Kalman filter implementation

        Input args:
        M: maximum count of objects in the scene (for fast computation?)
        P: covariance coef
        G_thresh: gating confidence threshold

        self.X: Object state (mx8) m: current object count (t)
        self.F: State transition matrix (8x8)
        self.Q: Measurement white noise matrix (8x8)
        self.H: Measurement matrix (4x8)
        self.P: Covariance matrix (8x8) state uncertainty
        self.R: Measurement noise matrix (4x4)
        self.K: Kalman gain (8x4)
        self.Z: Measurement (nx4) n: measured object count (t+1)
        
        Cell: 10, 120, 1, 30, R=[10, 10]
        Bat: 
        """
        self.G_thresh = G_thresh
        np.random.seed(random_seed)
        # self.X = None
        self.P = P * np.eye(4)
        self.F = self.init_kinematic_matrix()
        self.Q = np.random.normal(0.0, std_dev, (4, 4)) # std_dev = 0.01
        self.H = np.bmat([np.eye(2), np.zeros((2, 2))])#np.zeros((4, 8))
        self.R = np.diag([10, 10])
        self.K = np.zeros((4, 2))

        print("Init F:")
        print(self.F)

        print("Init P:")
        print(self.P)

        print("Init Q:")
        print(self.Q)

        print("Init H:")
        print(self.H)

        print("Init R:")
        print(self.R)

    def predict(self, X, P=None):
        """
        Prediction phase for Kalman filter.
        """
        if P is None:
            P = self.P

        X_pred = (X @ self.F.T) # (M x 8)
        P_pred = self.F @ P @ self.F.T + self.Q

        return np.array(X_pred), np.array(P_pred)
        
    def init_kinematic_matrix(self):
        F = np.eye(4)
        F[:2, -2:] = np.eye(2)
        return F

    def get_pred_state(self):
        return self.X

    def get_pred_cov(self):
        return self.P

    def convert_to_state(self, localization):
        """
        This function is used for converting the localization info
        for the first frame.

        localization: list of `State`. Object states in the current frame
        """
        
        print("Current number of objects in frame: %i" % len(localization))

        X = np.zeros((len(localization), 4)) # default value for nan: 
        for i, state in enumerate(localization):
            arr = state.to_array()
            X[i, :] = np.concatenate([arr[:2], arr[4:6]])
        return X

    def convert_to_measurement(self, localization):
        """
        This function is used for converting the localization info
        into measurement matrix for other frames

        localization: list of `State`. Object states in the current frame
        """

        print("Current number of objects in frame: %i" % len(localization))

        Z = np.zeros((len(localization), 2)) # default value for nan: 
        for i, state in enumerate(localization):
            Z[i, :] = state.to_array()[:2]
        return Z

    def gating(self, X_pred, P_pred, Z):
        
        object_graph = np.zeros((len(Z), len(X_pred)))
        # object_edge_cost = np.zeros((len(Z), self.M))

        self.S = self.H @ P_pred @ self.H.T + self.R
        
        conf_list = []
        for i, z in enumerate(Z):
            Y = z - (X_pred @ self.H.T) # z:(1x4) broadcasted; Y:(nX x 4) (residual)
            confidence = np.diag(Y @ np.linalg.pinv(self.S) @ Y.T) # ellipsoidal confidence score
            confidence = np.clip(confidence, -10.0, 10000.)
            conf_list.append("%.2f" % confidence.min())
            # print(confidence.min())
            G = (confidence < self.G_thresh).astype(np.float32) # G:(1 x nX)
            object_graph[i, :] = G
            # object_edge_cost[i, :] = G * confidence
        print("confidence:", ", ".join(conf_list))  
        # print("Object linkage graph:")
        # print(object_graph[:, :10])
        # print("Object edge cost:")
        # print(object_edge_cost[:, :10])

        return object_graph

    def update(self, X_pred, P_pred, Y):

        self.K = P_pred @ self.H.T @ np.linalg.pinv(self.S)
        print("Update", X_pred.shape, Y.shape, self.K.T.shape)
        X = X_pred + Y @ self.K.T
        # P = (np.identity(4) - self.K @ self.H) @ P_pred
        # P = P_pred - self.K @ self.H @ P_pred
        #(I-KH)P(I-KH)' + KRK'
        I = np.identity(4)
        I_KH =  (I - self.K @ self.H)
        P = I_KH @ P_pred @ I_KH.T + self.K @ self.R @ self.K.T

        return np.array(X), np.array(P)


    def calc_residual(self, X_pred, Z, Y_mask=None):
        assert(len(Z) == len(X_pred))
        Y = Z - (X_pred @ self.H.T)
        return np.array(Y)

    def visualize_object_graph(self, frame, object_graph, X_pred, Z, deleted_obj_ids={}):

        for z, linkages in zip(Z, object_graph):

            from_coord = (int(z[0]), int(z[1]))
            cv2.circle(frame, from_coord, 1, (0, 0, 255), -1)
            for i, linkage in enumerate(linkages):
                if i in deleted_obj_ids:
                    continue
                if linkage > 0.9:
                    to_coord = (int(X_pred[i][0]), int(X_pred[i][1]))
                    cv2.circle(frame, to_coord, 1, (255, 0, 0), -1)
                    cv2.line(frame, from_coord, to_coord, (255, 0, 0), 1)
                # break


if __name__ == "__main__":
    data = CellDataLoader("../data/cell/CS585-Cells/", 2, DEBUG=False)
    # data = BatDataLoader("../data/bats/CS585-BatImages/", 2, DEBUG=False)
    kf = KalmanFilter()
    X = kf.convert_to_state(data.localization[0])
    X_n, P = kf.predict(X)
    # print("X_pred", X)
    # print("P_pred", P)

    gnnsf = GNNSF()

    Z_prev = None

    deleted_obj_ids = {}
    for frame_id, (loc_data, frame) in enumerate(zip(data.localization[1:], data.images[1:])):

        if frame_id == 0:
            frame_init = data.images[0].copy()            
            for obj_id, x in enumerate(X):
                cv2.circle(frame_init, get_coord(x), 2, (255,0,0), -1) #or
                cv2.putText(frame_init, str(obj_id), get_coord(x), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0))
                cv2.imshow("Init", frame_init)
        else:
            cv2.imshow("Prev", data.images[frame_id-1])

        # Prediction phase
        X_pred, P_pred = kf.predict(X, P)
        Z = kf.convert_to_measurement(loc_data) # (Mx4)

        # Gating phase: remove linkages w/t lower confidence
        #               Set G_thresh to float('inf') if we don't want this step
        object_graph = kf.gating(X_pred, P_pred, Z)

        frame_debug = frame.copy()
        kf.visualize_object_graph(frame_debug, object_graph, X_pred, Z, deleted_obj_ids)
        cv2.imshow("object_graph", frame_debug)

        # Data Association phase: 
        # allignment, isolated_z, isolated_x = gnnsf.greedy_associate(object_graph, X_pred, Z, deleted_obj_ids)
        # print("Allignment", allignment)
        # print("iso_X", len(isolated_x), isolated_x)
        # print("iso_Z", len(isolated_z), isolated_z)
        allignment, isolated_z, isolated_x = gnnsf.hungarian_associate(object_graph, X_pred, Z, deleted_obj_ids)
        

        frame_obs = frame.copy()
        for obj_id, z in enumerate(Z):
            cv2.circle(frame_obs, get_coord(z), 2, (255,0,0), -1) #or
            cv2.putText(frame_obs, str(obj_id), get_coord(z), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0))

        # Managing object creation/deletion

        Y_mask = np.ones((len(allignment), Z.shape[1]))

        if len(isolated_z) > 0: # observed new object in current frame
            print("Detected %i new objects!" % len(isolated_z))
            # create dummy rows for X (new detections)
            X_pred = np.concatenate([X_pred, np.zeros((len(isolated_z), X_pred.shape[1]))])
            X = np.concatenate([X, np.zeros((len(isolated_z), X.shape[1]))])
            for x_id in isolated_z.keys():
                print("Concating:", np.bmat([Z[x_id], [0,0]]))
                X_pred[allignment[x_id], :] = np.bmat([Z[x_id], [0,0]])
                X[allignment[x_id], :] = np.bmat([Z[x_id], [0,0]])
                Y_mask[allignment[x_id], :] = np.zeros((1, Y_mask.shape[1]))
        print(X.shape, X_pred.shape)
        
        if len(isolated_x) > 0:
            print("Missing %i objects!" % len(isolated_x))
            # TODO: create HalfDead class for the missing object
            #       delete it when miss detecting for K frames
            for x_id in isolated_x.keys():
                deleted_obj_ids[x_id] = True

            # create zero-mask for residual when missing detections
            for x_id in isolated_x.keys():
                Y_mask[x_id, :] = np.zeros((1, Y_mask.shape[1]))


        # Re-alligning Z matrix
        nZ, nX = len(Z), len(X_pred)
        print("nZ", nZ)
        print("nX", nX)
        if nZ >= len(allignment):
            indexes = sorted(list(enumerate(allignment)), key=lambda x: x[1])
            indexes = [k for k, v in indexes]
            print(indexes)
            Z = Z[indexes, :]
        else:
            # create dummy rows for Z (miss detections)
            Z = np.concatenate([Z, np.zeros((len(allignment)-nZ, Z.shape[1]))])

            # indexes = np.arange(len(allignment))[allignment]
            indexes = sorted(list(enumerate(allignment)), key=lambda x: x[1])
            indexes = [k for k, v in indexes]
            Z = Z[indexes, :]


        # Calculate residuals
        Y = kf.calc_residual(X_pred, Z)
        Y = Y * Y_mask
        print("Residual", Y.shape)
        print("X_pred", X_pred.shape)
        print("Z", Z.shape)
        print("deleted", deleted_obj_ids)

        # Update:
        X_n, P_n = kf.update(X_pred, P_pred, Y)
        
        # Visualization
        drawn_id = {}
        # show creation
        for observer_id in isolated_z.keys():
            obj_id = allignment[observer_id]
            frame = cv2.circle(frame, get_coord(X_n[obj_id]), 1, (0,165,255), -1) #or
            frame = cv2.putText(frame, str(obj_id), get_coord(X_n[obj_id]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
            drawn_id[obj_id] = True

        # show missing
        # for obj_id in isolated_x.keys():
        #     frame = cv2.circle(frame, get_coord(X_n[obj_id]), 1, (255,102,178), -1) #pur
        #     frame = cv2.putText(frame, str(obj_id), get_coord(X_n[obj_id]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
        #     drawn_id[obj_id] = True

        # Tracking
        for obj_id in range(len(Z)):
            if obj_id in drawn_id:
                continue
            if obj_id in deleted_obj_ids:
                continue
            
            frame = cv2.circle(frame, get_coord(X[obj_id]), 1, (255,0,0), -1) #blue
            frame = cv2.circle(frame, get_coord(X_pred[obj_id]), 1, (0,255,0), -1) #green
            frame = cv2.circle(frame, get_coord(Z[obj_id]), 1, (0,255,255), -1) # yellow
            frame = cv2.circle(frame, get_coord(X_n[obj_id]), 1, (0,0,255), -1) # red
            frame = cv2.putText(frame, str("(%.2f, %.2f)" % (X_n[obj_id][2], X_n[obj_id][3])), get_coord(X[obj_id]), cv2.FONT_HERSHEY_COMPLEX, 0.2, (255,0,0))
            print("Residual: %i" % obj_id, Y[obj_id])
            cv2.line(frame, get_coord(X[obj_id]), get_coord(Z[obj_id]), (255, 0, 0), 1)

            frame = cv2.putText(frame, str(obj_id), get_coord(X[obj_id]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
            # frame = cv2.putText(frame, str(obj_id), get_coord(X_pred[obj_id]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
            # frame = cv2.putText(frame, str(obj_id), get_coord(X_n[obj_id]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
            frame = cv2.putText(frame, str(obj_id), get_coord(Z[obj_id]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0))

        print("P", P_n)
        print("K", kf.K)

        

        X, P = X_n, P_n

        # Verify velocity estimation
        if Z_prev is not None:
            obj_cnt = min(len(Z), len(Z_prev), len(X_pred))
            velocity = Z[:obj_cnt] - Z_prev[:obj_cnt]
            for obj_id in range(obj_cnt):
                print(obj_id, X_pred[obj_id, -2:], velocity[obj_id])
            Z_prev = Z
        else:
            Z_prev = Z

        while True:
            cv2.imshow("Tracking", frame)
            cv2.imshow("Measure", frame_obs)
            # cv2.imshow("Measure_after", frame_obs_after)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
