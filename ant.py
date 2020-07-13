import random

import numpy as np

class Ant:

    def __init__(self, points, current_i):
        self.MIN_PHEROMONE = 0.00001
        self.ANGLE_WEIGHT = 0.5
        self.DIST_WEIGHT = 0.5
        self.ALPHA = 0.5
        self.BETA = 0.5
        self.points = points
        self.current_i = current_i
        self.prev_i = None
        self.candidate_row_i = []
    
    
    def construct_solutions(self, kd_tree, k, pheromone_matrix):
        i = 0 

        while True:
            current = self.points[self.current_i]
            # The first (& closest) point returned by kd_tree.query is itself
            dists, points_i = kd_tree.query(current, k + 1)
            dists, points_i = dists[1:].tolist(), points_i[1:].tolist()
            if self.prev_i in points_i:
                index = points_i.index(self.prev_i)
                del dists[index]; del points_i[index]
            
            pheromones = self.get_pheromones(dists, points_i, pheromone_matrix)
            heuristics = self.get_heuristics(dists, points_i)
            probs = self.get_probabilities(pheromones, heuristics)
            print(heuristics); print(probs); print()
            next_i = random.choices(points_i, probs, k=1)[0]

            self.candidate_row_i.append(next_i)
            self.prev_i = self.current_i
            self.current_i = next_i
            
            if i == 100:
                break
            else:
                i = i + 1

    
    def update_pheromones(self):
        pass


    def apply_local_search(self):
        pass


    def get_pheromones(self, dists, points_i, pheromone_matrix):
        pheromones = []
        for dist, i in zip(dists, points_i):
            key = tuple(sorted([self.current_i, i]))
            if key in pheromone_matrix:
                pheromones.append(pheromone_matrix.get(key))
            else:
                pheromones.append(self.MIN_PHEROMONE)
        return pheromones
    

    def get_heuristics(self, dists, points_i):
        # TODO: incorporate angles.
        dist_heuristics = [1 - (dist/max(dists)) for dist in dists]
        angle_heuristics = []
        for next_i in points_i:
            if self.prev_i == None:
                angle_heuristics.append(0)
            else:
                prev_point = self.points[self.prev_i]
                current_point = self.points[self.current_i]
                next_point = self.points[next_i]
                angle = self.get_abs_angle(prev_point, current_point, next_point) 
                angle_heuristics.append(angle/180)

        print(dist_heuristics);print(angle_heuristics);
        combine = lambda a, b: a * self.DIST_WEIGHT + b * self.ANGLE_WEIGHT
        heuristics = list(map(combine, dist_heuristics, angle_heuristics))
        return heuristics


    def get_abs_angle(self, p0, p1, p2):
        '''Returns smallest angle of two connected lines'''
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)
        angle = np.math.atan2(np.linalg.det([v0,v1]), np.dot(v0,v1))
        return abs(np.degrees(angle))


    def get_probabilities(self, pheromones, heuristics):
        prob_rule = lambda a, b: a**self.ALPHA * b**self.BETA
        numerators = list(map(prob_rule, pheromones, heuristics)) 
        denomenator = sum(numerators)
        probs = [numerator/denomenator for numerator in numerators]
        return probs
