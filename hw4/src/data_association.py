import math
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

class GNNSF:
    def __init__(self, cost_method="euclidean"):
        self.cost_method = cost_method

    def greedy_associate(self, object_graph, X_pred, Z, deleted_obj_ids=None):
        """
        object_graph: Bipartite graph
        X_pred: state prediction from prev frame
        Z: current frame measurements
        """

        nZ, nX = len(Z), len(X_pred)

        # Create edge cost
        edge_cost = -1 * np.ones((nZ, nX))
        for observer_id, linkages in enumerate(object_graph):
            for object_id, linkage in enumerate(linkages):
                if deleted_obj_ids is not None and object_id in deleted_obj_ids:
                    continue
                if linkage == 0.0:
                    continue

                if self.cost_method == "euclidean":
                    score = self.euclidean_score(Z[observer_id], X_pred[object_id])
                elif self.cost_method == "iou":
                    score = self.iou_score(Z[observer_id], X_pred[object_id])
                elif self.cost_method == "template_match":
                    score = self.template_match(Z[observer_id], X_pred[object_id])
                
                edge_cost[observer_id][object_id] = score

        # print(edge_cost[:, :len(Z)])

        assignment = [] # observer_id: object_id
        used_object_id = {}
        isolated_z = {}
        isolated_x = {}
        
        # Greedy approach start
        # debug_assign = []
        for observer_id, costs in enumerate(edge_cost):

            # Detect isolated node in bipartite graph
            # if np.count_nonzero(costs) == 0:
            if all(v == -1. for v in costs):
                # Assign it to a new object id
                
                assignment.append(nX)
                isolated_z[observer_id] = True
                # print("used", nX)
                used_object_id[nX] = True
                # debug_assign.append((nX, -1))
                nX += 1
            # Non-isolated node
            else:
                max_obj_id, max_cost = -1, -.1
                for i, cost in enumerate(costs):
                    # print(observer_id, i, cost, i in used_object_id)
                    if i not in used_object_id and cost > max_cost:
                        max_cost = cost
                        max_obj_id = i

                # Isolated anyways
                if max_obj_id == -1:
                    assignment.append(nX)
                    isolated_z[observer_id] = True
                    used_object_id[nX] = True
                    # print("isolate used", nX)
                    # debug_assign.append((nX, -1))
                    nX += 1
                else:    
                    assignment.append(max_obj_id)
                    # print("used", max_obj_id)
                    used_object_id[max_obj_id] = 1
                    # debug_assign.append((max_obj_id, max_cost))
                # print("temp debug", debug_assign)

        for i in range(edge_cost.shape[1]):
            # Detect isolated node in bipartite graph
            # print("edgecost", edge_cost[:, i])
            if all(v == -1. for v in edge_cost[:, i]) is True:
                isolated_x[i] = True

        # print("Debug assign", debug_assign)
        # print("Deleted obj", deleted_obj_ids)
        # When we miss detect objects, we create dummy allignments
        # and zero-mask those allignments when calculating residuals
        if nX > nZ:
            for i in range(nX):
                if i not in used_object_id:
                    # print("Appending", i)
                    assignment.append(i)
                    isolated_x[i] = True

        return assignment, isolated_z, isolated_x


    def hungarian_associate(self,):
        pass

    def template_match(self, z, x):
        pass

    def euclidean_score(self, z, x):
        dist = math.sqrt(float(z[0] - x[0])**2 + float(z[1] - x[1])**2)
        if dist == 0.0:
            score = float("inf")
        else:
            score = 1./dist
        # print(z, x[:2], score)
        return score

    def iou_score(self, z, x):
        box1 = BBox2D([x[0], x[1], x[2], x[3]]) # predicted next frame bbox using velocity
        box2 = BBox2D(z[:4])
        return jaccard_index_2d(box1, box2)
