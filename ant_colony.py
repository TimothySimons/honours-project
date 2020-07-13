import random

import numpy
import scipy.spatial
import matplotlib.pyplot as plt

import geo_shape
from ant import Ant

class AntColony:
    
    def __init__(self, points, n):
        self.points = points
        self.pheromone_matrix = {}
        self.kd_tree = scipy.spatial.KDTree(points)
        point_indices = [random.randint(0, len(points) - 1) for _ in range(n)]
        self.ants = [Ant(points, index) for index in point_indices]


    def find_rows(self):
        for ant in self.ants:         
            ant.construct_solutions(self.kd_tree, 5, self.pheromone_matrix)


    def plot(self):
        np_points = numpy.array(self.points)
        x = np_points[:,0]
        y = np_points[:,1]
        plt.scatter(x, y)


        for ant in self.ants:
            row_point_indices = ant.candidate_row_i
            row_points = [self.points[index] for index in row_point_indices]
            np_row_points = numpy.array(row_points)
            row_x = np_row_points[:,0]
            row_y = np_row_points[:,1]
            plt.plot(row_x, row_y, color='r')
        
        plt.show()    
