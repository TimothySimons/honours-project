import random

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

import analysis
import geo_shape
from ant import Ant

class AntColony:
    
    def __init__(self, points):
        self.orig_points = points
        self.orig_kd_tree = scipy.spatial.KDTree(points)
        self.points = np.copy(points)
        self.pheromone_matrix = {}
        self.candidate_rows = []
        self.end_points = []


    def main_loop(self, num_ants, num_elitists, num_iter):
        while self.points.size != 0:
            self.kd_tree = scipy.spatial.KDTree(self.points)
            global_best = []
            for _ in range(num_iter):
                ants = self.search(num_ants)
                global_best += ants
                ants.sort(reverse=True, key=self.criteria)
                global_best = global_best[:num_ants]
            
            global_best.sort(reverse=True, key=len)
            global_best = global_best[:num_elitists]
            self.eat(global_best)
        #    analysis.plot(self.orig_points, self.candidate_rows, self.end_points)
        final_rows = self.stitch2()
        end_pts = [[r[0], r[-1]] for r in final_rows]
        analysis.plot(self.orig_points, final_rows, end_pts)


    def eat(self, global_best):
        delete_indices = []
        for ant in global_best:
            candidate_row = np.take(self.points, ant.candidate_row_i, axis=0)
            self.candidate_rows.append(candidate_row)
            self.end_points.append([candidate_row[0], candidate_row[-1]])
            delete_indices += ant.candidate_row_i
        self.points = np.delete(self.points, delete_indices, axis=0)


    def criteria(self, ant):
        trail = self.get_pher_trail(ant.candidate_row_i)
        if len(trail) == 0:
            return 0
        else:
            return sum(trail)/len(trail)


    def search(self, n):
        points_i= [random.randint(0, len(self.points) - 1) for _ in range(n)]
        ants = [Ant(self.points, i, self.orig_kd_tree) for i in points_i]
        for ant in ants:        
            ant.construct_solutions(self.kd_tree, 3, self.pheromone_matrix)
        for ant in ants:
            ant.pheromone_update(self.pheromone_matrix, window=5, min_periods=1, 
                    center=True, win_type='triang')
        return ants


    def stitch2(self):
        end_pts = np.array([[row[0], row[-1]] for row in self.candidate_rows])
        exists = [True for _ in range(len(self.candidate_rows))]
        final_rows = []
        while any(exists):
            index = exists.index(True)
            row = [point.tolist() for point in self.candidate_rows[index]]
            self.add_connections(row, 0, exists, end_pts)
            self.add_connections(row, -1, exists, end_pts)
            final_rows.append(row)
            exists[index] = False
        return final_rows

    
    def add_connections(self, row, direction, exists, end_pts):
        next_index, next_row = self.find_connection(row, direction, exists, end_pts)
        if next_row is not None:
            next_row = [point.tolist() for point in next_row]
            if direction == 0:
                row[:0] = next_row
            else:
                row[len(row):] = next_row

            exists[next_index] = False
            self.add_connections(row, direction, exists, end_pts)


    def find_connection(self, row, direction, exists, end_pts):
        pair = [row[0], row[-1]]
        end_pt = pair[direction]
        dists, neighbours_i = self.orig_kd_tree.query(end_pt, 6)
        points = self.orig_points[neighbours_i]
        
        flat = end_pts.reshape(len(end_pts)*2, 2)
        bool_f = lambda p: True if (flat == p).any() and (pair != p).all() else False
        bool_mask = list(map(bool_f, points))
        traversable = points[bool_mask]
        dists = dists[bool_mask]
        
        if traversable.size == 0:
            return None, None

        next_point = traversable[np.argmin(dists)]
        next_index = flat.tolist().index(next_point.tolist())//2
        next_row = self.candidate_rows[next_index]
        
        if not exists[next_index]:
            return None, None

        if (next_row[0] == next_point).all() and direction == 0:
            next_row = next_row[::-1]
        elif (next_row[-1] == next_point).all() and direction == -1:
            next_row = next_row[::-1]

        return next_index, next_row
            


    def get_pher_trail(self, points_i):
        pheromone_trail = []
        for index in range(len(points_i)-1):
            point_i = points_i[index]
            point_j = points_i[index + 1]
            key = tuple(sorted([point_i, point_j]))
            pheromone_value = self.pheromone_matrix[key]
            pheromone_trail.append(pheromone_value)
        return pheromone_trail


#TODO: elitism
#TODO: global search that connects all chosen rows (end = start & nearest next) 
#TODO: if criteria or pheromone update incorporate length, then the model will perform
#      poorly in the case of mixed row length
#TODO: maybe when removing rows, sort by length
