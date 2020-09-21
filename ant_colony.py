import functools
import random
import statistics
import sys
import time

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

import geo_shape
from ant import Ant

class AntColony:
    """The Ant Colony is the point of mediation between all ants involved"""
    
    def __init__(self, points, config):
        self.MIN_ANGLE = config['min_angle']    # required for stitching
        self.EVAP_RATE = config['evap_rate']
        self.WINDOW = config['pher_update']['window']
        self.WIN_TYPE = config['pher_update']['win_type']
        self.MIN_PHER = 0.00001 
        self.config = config
        self.orig_points = points
        self.orig_kd_tree = scipy.spatial.KDTree(points)
        self.points_orig_i = np.array([i for i in range(len(points))])
        self.points = np.copy(points)
        self.pheromone_matrix = {}
        self.candidate_rows = []
        self.end_points = []

        self.best_heuristics = []


    def timer(func):
        """Print the runtime of the decorated function."""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
            sys.stdout.flush()
            return value
        return wrapper_timer


    @timer
    def main_loop(self, num_ants, num_elitists, num_iter):
        while self.points.size != 0:
            self.kd_tree = scipy.spatial.KDTree(self.points)
            global_best = []
            for _ in range(num_iter):
                ants = self.search(num_ants)
                global_best += ants
                global_best = sorted(global_best, reverse=True, key=self.criteria)
                global_best = global_best[:num_ants]

            global_best.sort(reverse=True, key=lambda x: len(x.candidate_row_i))
            global_best = global_best[:num_elitists]
            self.eat(global_best)
        final_rows = self.stitch()
        return final_rows


    def eat(self, global_best):
        delete_indices = []
        for ant in global_best:
            if not np.in1d(ant.candidate_row_i, delete_indices).any():
                candidate_row = np.take(self.points, ant.candidate_row_i, axis=0)
                self.candidate_rows.append(candidate_row)
                self.end_points.append([candidate_row[0], candidate_row[-1]])
                delete_indices += ant.candidate_row_i
        self.points_orig_i = np.delete(self.points_orig_i, delete_indices, axis=0)
        self.points = np.delete(self.points, delete_indices, axis=0)


    def criteria(self, ant):
        heuristics = ant.candidate_heuristics
        if len(heuristics) == 0:
            return 0
        else:
            return sum(heuristics)


    def search(self, n):
        points_i = [random.randint(0, len(self.points) - 1) for _ in range(n)]
        ants = []
        for i in points_i:
            config = self.config['ant']
            ant = Ant(self.points, i, self.orig_kd_tree, self.points_orig_i, config)
            ants.append(ant)
        for ant in ants:        
            ant.construct_solution(self.kd_tree, self.pheromone_matrix)
        for key, pher_value in self.pheromone_matrix.items():
            self.pheromone_matrix[key] = min(self.MIN_PHER, self.EVAP_RATE * pher_value)
        for ant in ants:
            ant.pheromone_update(self.pheromone_matrix, min_periods=1, center=True,
                    window=self.WINDOW, win_type=self.WIN_TYPE)
        return ants


    @timer
    def stitch(self):
        end_pts = np.array([[row[0], row[-1]] for row in self.candidate_rows])
        exists = [True for _ in range(len(self.candidate_rows))]
        final_rows = []
        while any(exists): 
            index = exists.index(True)
            exists[index] = False
            row = [point.tolist() for point in self.candidate_rows[index]]
            self.add_connections(row, 0, exists, end_pts)
            self.add_connections(row, -1, exists, end_pts)
            final_rows.append(row)
        return final_rows

    
    def add_connections(self, row, direction, exists, end_pts):
        next_index, next_row = self.find_connection(row, direction, exists, end_pts)
        if next_row is not None:
            next_row = [point.tolist() for point in next_row]
            if direction == 0 and self.valid_angle(row[::-1], next_row[::-1]):
                row[:0] = next_row
                exists[next_index] = False
                self.add_connections(row, direction, exists, end_pts)
            elif self.valid_angle(row, next_row):
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


    def valid_angle(self, current_row, next_row):
        p1 = current_row[-1]
        p2 = next_row[0]

        if len(current_row) > 1:
            p0 = current_row[-2]
            if self.get_abs_angle(p0, p1, p2) < 144:
                return False

        if len(next_row) > 1:
            p3 = next_row[1]
            if self.get_abs_angle(p1, p2, p3) < 144:
                return False
        
        return True
            

    def get_abs_angle(self, p0, p1, p2):
        """Returns the angle at vertex p1 enclosed by rays p0 p1 and p1 p2."""
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)
        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return abs(np.degrees(angle))


    def get_pher_trail(self, points_i):
        pheromone_trail = []
        for index in range(len(points_i)-1):
            point_i = points_i[index]
            point_j = points_i[index + 1]
            key = tuple(sorted([point_i, point_j]))
            pheromone_value = self.pheromone_matrix[key]
            pheromone_trail.append(pheromone_value)
        return pheromone_trail
