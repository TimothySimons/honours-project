import random

import numpy as np
import pandas as pd


class Ant:
    """The ants used in this modified ACO algorithm are stochastic solution
    construction procedures that probabilistically build orchard rows by iteratively
    adding edge components to partial candidate rows.

    This construction procedure takes into account:
    (i)  heuristic information in the form of distance and angle information.
    (ii) pheromone trails which change dynamically at run-time to reflect the agents’
         acquired search experience.
    """ 


    def __init__(self, points, current_i, orig_kd_tree, points_orig_i, config):
        """Constructor for ant objects."""
        self.K_NEIGHBOURS = config['neighbours']
        self.MAX_DIST = max(orig_kd_tree.query(points[current_i], config['allowable'])[0])
        self.MIN_ANGLE = config['min_angle']
        self.DIST_WEIGHT = config['dist_weight']
        self.ANGLE_WEIGHT = config['angle_weight']
        self.ALPHA = config['alpha']
        self.BETA = config['beta']
        self.MIN_PHER = config['min_pheromone']

        self.orig_kd_tree = orig_kd_tree
        self.points_orig_i = points_orig_i
        self.points = points
        self.candidate_row_i = [current_i]
        self.candidate_heuristics = []
        self.candidate_dists = []
        self.candidate_angles = []


    def construct_solution(self, kd_tree, pher_matrix, mm_dist, mm_angle):
        """Constructs a candidate row.

        The construction procedure starts at a given point and iteratively adds edges to
        this point (starting in one direction followed by the other).
        """ 
        self.construct_partial(kd_tree, self.K_NEIGHBOURS, pher_matrix, mm_dist, 
                mm_angle)
        self.candidate_row_i.reverse()
        self.construct_partial(kd_tree, self.K_NEIGHBOURS, pher_matrix, mm_dist, mm_angle)


    def construct_partial(self, kd_tree, k, pher_matrix, mm_dist, mm_angle):
        """Constructs partial candidate row."""
        while True:
            dists, neighbours_i = self.unexplored_nearest(kd_tree, k)
            if neighbours_i.size == 0:
                break
            
            angles = self.get_neighbour_angles(neighbours_i)
            pareto_i, dist_hs, angle_hs = self.pareto_heuristics(dists, angles, 
                    mm_dist, mm_angle, neighbours_i)

            combine = lambda dh, ah: dh*self.DIST_WEIGHT + ah*self.ANGLE_WEIGHT
            heuristics = np.array(list(map(combine, dist_hs, angle_hs)))
            pheromones = self.get_pheromones(pareto_i, pher_matrix)

            probs = self.get_probabilities(pheromones, heuristics)
            next_i = random.choices(pareto_i, probs, k=1)[0]

            a_index = pareto_i.tolist().index(next_i)
            d_index = neighbours_i.tolist().index(next_i)
            angle, dist = angles[a_index], dists[d_index]
            if angle < self.MIN_ANGLE or dist > self.MAX_DIST:
                break

            self.candidate_dists.append(dist)
            self.candidate_angles.append(angle)
            index = pareto_i.tolist().index(next_i)
            self.candidate_heuristics.append(heuristics[index])
            self.candidate_row_i.append(next_i)



    def unexplored_nearest(self, kd_tree, k):
        """Returns the unexplored k closest neighbours and distances to the current point."""
        current = self.points[self.candidate_row_i[-1]]
        dists, neighbours_i = kd_tree.query(current, k)
        unexplored = lambda i, d: i not in self.candidate_row_i and d != float('inf')
        unexplored_mask = list(map(unexplored, neighbours_i, dists))
        dists, neighbours_i = dists[unexplored_mask], neighbours_i[unexplored_mask]
        return dists, neighbours_i


    def pareto_heuristics(self, dists, angles, mm_dist, mm_angle, neighbours_i):
        """Calculates pareto heuristics of traversable neighbours of the candidate row."""
        neighbours = self.points[neighbours_i]
        p1 = self.points[self.candidate_row_i[-1]]
        if len(self.candidate_row_i) > 1:
            p0 = self.points[self.candidate_row_i[-2]]
        else:
            p0 = None

        scale_dist = lambda d: 1 - ((d - mm_dist[0]) / (mm_dist[1] - mm_dist[0]))
        scale_angle = lambda a: (a - mm_angle[0]) / (mm_angle[1] - mm_angle[0])
        bandpass = lambda x: max(min(1, x), 0)
        dist_heuristics = np.array(list(map(scale_dist, dists)))
        angle_heuristics = np.array(list(map(scale_angle, angles)))
        dist_heuristics = np.array(list(map(bandpass, dist_heuristics)))
        angle_heuristics = np.array(list(map(bandpass, angle_heuristics)))
        
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


    def pheromone_update(self, pheromone_matrix, **r_kwargs):
        """Updates global pheromone matrix with pheromones of this ant."""
        df = pd.DataFrame(self.candidate_heuristics)
        if not df.empty:
            pheromone_updates = df.rolling(**r_kwargs).mean()[0].to_numpy()
            for index, pheromone_update in enumerate(pheromone_updates):
                i, j = self.candidate_row_i[index], self.candidate_row_i[index + 1]
                orig_i, orig_j = self.points_orig_i[i], self.points_orig_i[j]
                key = tuple(sorted([orig_i, orig_j]))
                if key in pheromone_matrix:
                    pheromone = pheromone_matrix[key]
                    pheromone_matrix[key] = pheromone + pheromone_update
                else:
                    pheromone_matrix[key] = pheromone_update


    def get_pheromones(self, points_i, pheromone_matrix):
        """Retrieves the total deposited pheromone values on the ants candidate_row."""
        current_i = self.candidate_row_i[-1]
        pheromones = []
        for i in points_i:
            orig_i, orig_j = self.points_orig_i[current_i], self.points_orig_i[i]
            key = tuple(sorted([orig_i, orig_j]))
            if key in pheromone_matrix:
                pheromones.append(pheromone_matrix[key])
            else:
                pheromones.append(self.MIN_PHER)
        return pheromones


    def get_neighbour_angles(self, neighbours_i):
        """Returns a list of neighbouring angles"""
        neighbours = self.points[neighbours_i]
        p0 = None
        if len(self.candidate_row_i) > 1:
            p0 = self.points[self.candidate_row_i[-2]]
        p1 = self.points[self.candidate_row_i[-1]]

        if p0 is None:
            return np.full(len(neighbours), 180)
        return np.array([self.get_abs_angle(p0, p1, p2) for p2 in neighbours])


    def get_abs_angle(self, p0, p1, p2):
        """Returns the angle at vertex p1 enclosed by rays p0 p1 and p1 p2."""
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)
        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return abs(np.degrees(angle))


    def get_probabilities(self, pheromones, heuristics):
        """Calculates the transition probability distribution for a set of points."""
        prob_rule = lambda a, b: a**self.ALPHA * b**self.BETA
        numerators = list(map(prob_rule, pheromones, heuristics))
        denomenator = sum(numerators)
        probs = [numerator/denomenator for numerator in numerators]
        return probs
