import math
import numpy as np
import sys

from scipy.optimize import linear_sum_assignment

np.set_printoptions(threshold=sys.maxsize)

class GNNSF:
    def __init__(self, cost_method="euclidean"):
        self.cost_method = cost_method

    def get_edge_cost_matrix(self, object_graph, X_pred, Z, deleted_obj_ids, inv=False):
        nZ, nX = len(Z), len(X_pred)

        # Create edge cost
        edge_cost = -1 * np.ones((nZ, nX)) if not inv else 1000000. * np.ones((nZ, nX))
        for observer_id, linkages in enumerate(object_graph):
            for object_id, linkage in enumerate(linkages):
                if deleted_obj_ids is not None and object_id in deleted_obj_ids:
                    continue
                if linkage == 0.0:
                    continue

                if self.cost_method == "euclidean":
                    score = self.euclidean_score(Z[observer_id], X_pred[object_id], inv)
                elif self.cost_method == "iou":
                    score = self.iou_score(Z[observer_id], X_pred[object_id])
                elif self.cost_method == "template_match":
                    score = self.template_match(Z[observer_id], X_pred[object_id])
                
                edge_cost[observer_id][object_id] = score

        return edge_cost

    def get_edge_cost_matrix_2(self, object_graph, X_pred, Z, deleted_obj_ids):
        nZ, nX = len(Z), (len(X_pred) - len(deleted_obj_ids))

        id_mapping = []
        for obj_id in range(len(X_pred)):
            if obj_id not in deleted_obj_ids:
                id_mapping.append(obj_id)
        assert(len(id_mapping) == nX)

        # inv_id_mapping = {obj_id: i for i, obj_id in enumerate(id_mapping)}

        # Create edge cost
        edge_cost = 1000000. * np.ones((nZ, nX))
        for observer_id in range(len(object_graph)):
            for col_id, obj_id in enumerate(id_mapping):
                linkage = object_graph[observer_id][obj_id]
                if linkage == 0.0:
                    continue

                if self.cost_method == "euclidean":
                    score = self.euclidean_score(Z[observer_id], X_pred[obj_id], True)
                edge_cost[observer_id][col_id] = score
        return edge_cost, id_mapping

    def greedy_associate(self, object_graph, X_pred, Z, deleted_obj_ids=None):
        """
        object_graph: Bipartite graph
        X_pred: state prediction from prev frame
        Z: current frame measurements
        """        
        edge_cost = self.get_edge_cost_matrix(object_graph, X_pred, Z, deleted_obj_ids)        

        assignment = [] # observer_id: object_id
        used_object_id = {}
        isolated_z = {}
        isolated_x = {}
        nZ, nX = len(Z), len(X_pred)

        # Greedy approach start
        # debug_assign = []
        total_cost = 0.0
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
                    total_cost += max_cost
                    # debug_assign.append((max_obj_id, max_cost))
                # print("temp debug", debug_assign)

        for i in range(edge_cost.shape[1]):
            # Detect isolated node in bipartite graph
            # print("edgecost", edge_cost[:, i])
            if all(v == -1. for v in edge_cost[:, i]) is True:
                isolated_x[i] = True

        # When we miss detect objects, we create dummy allignments
        # and zero-mask those allignments when calculating residuals
        if nX > nZ:
            for i in range(nX):
                if i not in used_object_id:
                    # print("Appending", i)
                    assignment.append(i)
                    isolated_x[i] = True

        print("Greedy approach total cost: ", total_cost)
        return assignment, isolated_z, isolated_x


    def hungarian_associate(self, object_graph, X_pred, Z, deleted_obj_ids=None):

        edge_cost, id_mapping = self.get_edge_cost_matrix_2(object_graph, X_pred, Z, deleted_obj_ids)
        # print("Edgecost", edge_cost.shape, edge_cost)
        row_ind, col_ind = linear_sum_assignment(edge_cost)

        # Handle default distance situation:
        for i, (row_id, col_id) in enumerate(zip(row_ind, col_ind)):
            if edge_cost[row_id][col_id] >= 999999.:
                row_ind = np.delete(row_ind, i)
                col_ind = np.delete(col_ind, i) #col_ind.delete(i)

        col_ind = [id_mapping[i] for i in col_ind]

        nZ, nX = len(Z), len(X_pred)
        assignment = []
        # isolated z
        isolated_z = {}
        used_row_id = {ind: col_ind[i] for i, ind in enumerate(row_ind)}
        used_obj_id = {}
        for i in range(nZ):
            if i not in used_row_id:
                isolated_z[i] = True
                assignment.append(nX)
                used_obj_id[nX] = True
                nX += 1
            else:
                assignment.append(used_row_id[i])
                used_obj_id[used_row_id[i]] = True
        
        # isolated x
        isolated_x = {}
        used_col_id = {i: True for i in col_ind}
        for i in range(len(X_pred)):
            if i not in used_col_id and i not in used_obj_id:
                isolated_x[i] = True
                assignment.append(i)

        assert(len(row_ind) == len(col_ind))

        print("=====Hungarian!!======")
        print("row_ind", row_ind)
        print("col_ind", col_ind)
        print("isolated_x", isolated_x)
        print("isolated_z", isolated_z)
        print("assignment", assignment)

        return assignment, isolated_z, isolated_x

    def template_match(self, z, x):
        pass

    def euclidean_score(self, z, x, inv=False):
        dist = math.sqrt(float(z[0] - x[0])**2 + float(z[1] - x[1])**2)
        if not inv:
            if dist == 0.0:
                score = float("inf")
            else:
                score = 1./dist
        else:
            score = dist
        # print(z, x[:2], score)
        return score

    def iou_score(self, z, x):
        box1 = BBox2D([x[0], x[1], x[2], x[3]]) # predicted next frame bbox using velocity
        box2 = BBox2D(z[:4])
        return jaccard_index_2d(box1, box2)
