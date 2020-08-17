import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial

import analysis

class Ant:

    def __init__(self, points, current_i, orig_kd_tree):
        self.MIN_PHEROMONE = 0.00001 
        self.MIN_ANGLE_H = 0.80
        self.ALLOWABLE = 6
        self.DIST_WEIGHT = 0.8
        self.ANGLE_WEIGHT = 0.2
        self.ALPHA = 0.9
        self.BETA = 0.1

        self.points = points
        self.candidate_row_i = [current_i]
        self.candidate_heuristics = []
        self.orig_kd_tree = orig_kd_tree


    def construct_solutions(self, kd_tree, k, pheromone_matrix):
        self.half(kd_tree, k, pheromone_matrix)
        self.candidate_row_i.reverse()
        self.half(kd_tree, k, pheromone_matrix)


    def half(self, kd_tree, k, pheromone_matrix):

        while True:
            dists, neighbours_i = self.unexplored_nearest(kd_tree, k)
            if neighbours_i.size == 0: 
                break

            pareto_i, dist_hs, angle_hs = self.pareto_heuristics(dists, neighbours_i)
            combine = lambda dh, ah: dh * self.DIST_WEIGHT + ah * self.ANGLE_WEIGHT
            heuristics = np.array(list(map(combine, dist_hs, angle_hs)))
            pheromones = self.get_pheromones(dists, pareto_i, pheromone_matrix)

            probs = self.get_probabilities(pheromones, heuristics)
            next_i = random.choices(pareto_i, probs, k=1)[0]

            a_index = pareto_i.tolist().index(next_i)
            d_index = neighbours_i.tolist().index(next_i)
            next_point = self.points[next_i]
            if self.terminate(next_point, angle_hs[a_index], dists[d_index]): 
                break

            index = pareto_i.tolist().index(next_i)
            self.candidate_heuristics.append(heuristics[index])
            self.candidate_row_i.append(next_i)
        


    def unexplored_nearest(self, kd_tree, k):
        """Returns the unexplored k closest neighbours and distances to the current point."""
        current = self.points[self.candidate_row_i[-1]]
        dists, neighbours_i = kd_tree.query(current, k)
        unexplored = lambda i, d: i not in self.candidate_row_i and d != float('inf')
        unexplored_mask = np.array(list(map(unexplored, neighbours_i, dists)))
        dists, neighbours_i = dists[unexplored_mask], neighbours_i[unexplored_mask]
        return dists, neighbours_i


    def pareto_heuristics(self, dists, neighbours_i):
        """Calculates pareto heuristics of traversable neighbours of the candidate row."""
        neighbours = self.points[neighbours_i]
        p_1_i = self.candidate_row_i[-1]
        p_1 = self.points[p_1_i]
        if len(self.candidate_row_i) > 1:
            p_0_i = self.candidate_row_i[-2]
            p_0 = self.points[p_0_i]
        else:
            p_0 = None

        get_dist_heuristics = lambda d: 1 - (d/max(dists))
        get_angle_heuristics = lambda p_2: self.get_abs_angle(p_0, p_1, p_2)/180
        dist_heuristics = np.array(list(map(get_dist_heuristics, dists)))
        if p_0 is None:
            angle_heuristics = np.ones(len(neighbours))
        else:
            angle_heuristics = np.array(list(map(get_angle_heuristics, neighbours)))
        
        scores = [[d, a] for d, a in zip(dist_heuristics, angle_heuristics)]
        pareto_mask = self.pareto(np.array(scores))
        dist_heuristics = dist_heuristics[pareto_mask]
        angle_heuristics = angle_heuristics[pareto_mask]
        pareto_front_i = neighbours_i[pareto_mask]

        return pareto_front_i, dist_heuristics, angle_heuristics


    def pareto(self, scores):
        """Returns a boolean array indicating which points are Pareto efficient."""
        is_efficient = np.ones(scores.shape[0], dtype=bool)
        for i, score in enumerate(scores):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(scores[is_efficient] > score, axis=1)
                is_efficient[i] = True
        return is_efficient


    def terminate(self, next_point, angle_h, dist):
        dists, _ = self.orig_kd_tree.query(next_point, self.ALLOWABLE) 
        if angle_h < self.MIN_ANGLE_H or (dists < dist).all(): 
            return True
        return False


    def pheromone_update(self, pheromone_matrix, **r_kwargs):
        df = pd.DataFrame(self.candidate_heuristics)
        if not df.empty:
            pheromone_updates = df.rolling(**r_kwargs).mean()[0].to_numpy()
            for index, pheromone_update in enumerate(pheromone_updates):
                i, j = self.candidate_row_i[index], self.candidate_row_i[index + 1]
                key = tuple(sorted([i, j]))
                if key in pheromone_matrix:
                    pheromone = pheromone_matrix[key]
                    pheromone_matrix[key] = pheromone + pheromone_update
                else:
                    pheromone_matrix[key] = pheromone_update


    def get_pheromones(self, dists, points_i, pheromone_matrix):
        current_i = self.candidate_row_i[-1]
        pheromones = []
        for dist, i in zip(dists, points_i):
            key = tuple(sorted([current_i, i]))
            if key in pheromone_matrix:
                pheromones.append(pheromone_matrix[key])
            else:
                pheromones.append(self.MIN_PHEROMONE)
        return pheromones


    def get_abs_angle(self, p_0, p_1, p_2):
        """Returns the angle at vertex p_1 enclosed by rays p_0 p_1 and p_1 p_2.""" 
        v_0 = np.array(p_0) - np.array(p_1)
        v_1 = np.array(p_2) - np.array(p_1)
        angle = np.math.atan2(np.linalg.det([v_0, v_1]), np.dot(v_0, v_1))
        return abs(np.degrees(angle))


    def get_probabilities(self, pheromones, heuristics):
        """Calculates the transition probability distribution for a set of points."""
        prob_rule = lambda a, b: a**self.ALPHA * b**self.BETA
        numerators = list(map(prob_rule, pheromones, heuristics))
        denomenator = sum(numerators)
        probs = [numerator/denomenator for numerator in numerators]
        return probs

    def __len__(self):
        return len(self.candidate_row_i)

#TODO: improve local search
#TODO: add evaporation rate
