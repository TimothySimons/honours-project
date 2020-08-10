import random

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

import geo_shape
from ant import Ant

class AntColony:
    
    def __init__(self, points):
        self.points = points
        self.pheromone_matrix = {}
        self.candidate_rows = []
        self.kd_tree = scipy.spatial.KDTree(points)
    

    def main_loop(self, num_iter, num_ants):
        global_best = []
        for _ in range(num_iter):
            ants = self.search(num_ants)
            global_best += ants
            ants.sort(key=self.criteria)
            global_best = global_best[:num_ants]
        self.plot(global_best) #TODO


    def criteria(self, ant):
        trail = self.get_pher_trail(ant.candidate_row_i)
        return len(trail)


    def search(self, n):
        points_i= [random.randint(0, len(self.points) - 1) for _ in range(n)]
        ants = [Ant(self.points, self.pheromone_matrix, i) for i in points_i]
        for ant in ants:        
            ant.construct_solutions(self.kd_tree, 5, self.pheromone_matrix)
            ant.pheromone_update(window=5, min_periods=1, center=True, win_type='triang')
        return ants


    def get_pher_trail(self, points_i):
        pheromone_trail = []
        for index in range(len(points_i)-1):
            point_i = points_i[index]
            point_j = points_i[index + 1]
            key = tuple(sorted([point_i, point_j]))
            pheromone_value = self.pheromone_matrix[key]
            pheromone_trail.append(pheromone_value)
        return pheromone_trail


    def plot(self, ants):
        np_points = np.array(self.points)
        x = np_points[:,0]
        y = np_points[:,1]
        plt.scatter(x, y)
        for ant in ants:
            row_point_indices = ant.candidate_row_i
            row_points = [self.points[index] for index in row_point_indices]
            np_row_points = np.array(row_points)
            row_x = np_row_points[:,0]
            row_y = np_row_points[:,1]
            plt.plot(row_x, row_y, color='r')
        plt.show()    


    def plot_pheromones(self):
        np_points = np.array(self.points)
        x = np_points[:,0]
        y = np_points[:,1]
        plt.scatter(x, y)
        for key, pheromone in self.pheromone_matrix.items():
            point_0 = self.points[key[0]]
            point_1 = self.points[key[1]]
            np_points = np.array([point_0, point_1])
            x = np_points[:, 0]
            y = np_points[:, 1]
            plt.plot(x, y, color='r')
        plt.show()


