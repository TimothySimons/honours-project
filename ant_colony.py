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
            #analysis.plot(self.orig_points, self.candidate_rows, self.end_points)
        self.stitch()


    def eat(self, global_best):
        delete_indices = []
        for ant in global_best:
            candidate_row = np.take(self.points, ant.candidate_row_i, axis=0)
            self.candidate_rows.append(candidate_row)
            #TODO: currently doesn't handle for rows of length 1
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

    
    def stitch(self):
        self.candidate_rows.sort(key=len)
        end_pts = np.array([[row[0], row[-1]] for row in self.candidate_rows])
        
        connections = []
        for index, pair in enumerate(end_pts):
            head, tail = pair[0], pair[-1]
            h_connect = self.find_connection(end_pts, head, pair)
            t_connect = self.find_connection(end_pts, tail, pair)
            if h_connect is not None:
                connections.append([head, h_connect])
            if t_connect is not None:
                connections.append([tail, t_connect])
        analysis.plot_next(self.orig_points, self.candidate_rows, end_pts, connections)
            


    def find_connection(self, end_pts, end_pt, pair):
        dists, neighbours_i = self.orig_kd_tree.query(end_pt, 6)
        points = self.orig_points[neighbours_i]
        
        flat = end_pts.reshape(len(end_pts)*2, 2)
        bool_f = lambda p: True if (flat == p).any() and (pair != p).all() else False
        bool_mask = list(map(bool_f, points))
        traversable = points[bool_mask]
        dists = dists[bool_mask]
        
        if traversable.size != 0:
            return traversable[np.argmin(dists)]
        else:
            return None

        
        

        


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
