import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

import geo_shape
from ant import Ant


def construct_ground_truth(row_points):
    kd_tree = scipy.spatial.KDTree(row_points)
    row_i = [random.randrange(0,len(row_points) - 1)]
    construct_partial(row_i, row_points, kd_tree)
    row_i.reverse()
    construct_partial(row_i, row_points, kd_tree)
    row = [row_points[i] for i in row_i]
    return row


def construct_partial(row_i, row_points, kd_tree, k=10):
    while(True):
        current = row_points[row_i[-1]]
        dists, neighbours_i = kd_tree.query(current, k)

        unexplored = lambda i, d: i not in row_i and d != float('inf')
        unexplored_mask = list(map(unexplored, neighbours_i, dists))
        dists, neighbours_i = dists[unexplored_mask], neighbours_i[unexplored_mask]

        straight = lambda i: valid_angle(row_points, row_i, i)
        straight_mask = list(map(straight, neighbours_i))
        dists, neighbours_i = dists[straight_mask], neighbours_i[straight_mask]

        if neighbours_i.size == 0:
            break
        else:
            next_i = neighbours_i[0]
            row_i.append(next_i)


def valid_angle(row_points, row_i, next_i):
    row = [row_points[i] for i in row_i]
    p2 = row_points[next_i]
    if len(row) > 1:
        p0 = row[-2]
        p1 = row[-1]
        angle = get_abs_angle(p0, p1, p2)
        if angle < 90:
            return False
    return True


def get_abs_angle(p_0, p_1, p_2):
    """Returns the angle at vertex p_1 enclosed by rays p_0 p_1 and p_1 p_2."""
    v_0 = np.array(p_0) - np.array(p_1)
    v_1 = np.array(p_2) - np.array(p_1)
    angle = np.math.atan2(np.linalg.det([v_0, v_1]), np.dot(v_0, v_1))
    return abs(np.degrees(angle))


def construct_edge_set(model_rows, true_rows):
    #import pdb; pdb.set_trace()
    current_index = 0
    for m_row in model_rows:
        for t_row in true_rows:
            current_index = assign_indices(m_row, t_row, current_index)
    print(current_index)
    m_edge_set = rows_to_edge_set(model_rows)
    t_edge_set = rows_to_edge_set(true_rows)
    return m_edge_set, t_edge_set


def assign_indices(model_row, true_row, current_index):
    for m_index in range(len(model_row)):
        for t_index in range(len(true_row)):
            m_point = model_row[m_index]
            t_point = true_row[t_index]
            if type(m_point) is tuple or type(t_point) is tuple:
                continue
            elif np.allclose(m_point, t_point, rtol=0, atol=1e-3):
                model_row[m_index] = (current_index, m_point)
                true_row[t_index] = (current_index, t_point)
                current_index += 1
    return current_index


def rows_to_edge_set(rows):
    edge_set = set()
    for row in rows:
        for i in range(len(row)-1):
            j = i + 1
            if type(row[i]) is not tuple or type(row[j]) is not tuple:
                edge_set.add((None, None))
            else:
                point_i = row[i][0]
                point_j = row[j][0]
                edge_set.add(tuple(sorted([point_i, point_j])))
    return edge_set


def precision(model_edge_set, true_edge_set):
    true_positive = 0
    false_positive = 0
    for edge in model_edge_set:
        if edge in true_edge_set and edge is not (None, None):
            true_positive += 1
        else:
            false_positive += 1

    return true_positive / (true_positive + false_positive)
    


def recall(model_edge_set, true_edge_set):
    true_positive = 0
    for edge in model_edge_set:
        if edge in true_edge_set and edge is not (None, None):
            true_positive += 1

    false_negative = 0
    for edge in true_edge_set:
        if edge not in model_edge_set or edge is (None, None):
            false_negative += 1

    return true_positive / (true_positive + false_negative)


def plot_rows(points, rows):
    np_points = np.array(points)
    x = np_points[:,0]
    y = np_points[:,1]
    plt.scatter(x, y)

    for row in rows:
        np_row = np.array(row)
        row_x = np_row[:,0]
        row_y = np_row[:,1]
        plt.plot(row_x, row_y, color='r')
    plt.show()   


def plot_trails(points, rows_i, pheromone_matrix):
    fig, axs = plt.subplots(2)
    fig.suptitle('Candidate rows and their pheromone values')
    np_points = np.array(points)
    x = np_points[:,0]
    y = np_points[:,1]
    axs[0].scatter(x, y, color='gray')
    
    pheromone_trails = []
    for num, row_i in enumerate(rows_i):
        pheromone_trail = []
        for i in range(len(row_i) - 1):
            point_i = row_i[i]
            point_j = row_i[i + 1]
            key = tuple(sorted([point_i, point_j]))
            pheromone_value = pheromone_matrix[key]
            pheromone_trail.append(pheromone_value)
        pheromone_trails.append(pheromone_trail)

    for row_i in rows_i:
        row_points = [points[index] for index in row_i]
        np_row_points = np.array(row_points)
        row_x = np_row_points[:,0]
        row_y = np_row_points[:,1]
        axs[0].plot(row_x, row_y)

    for trail in pheromone_trails:
        axs[1].plot(trail)

    plt.show()    


def plot_precision(points ,rows, true_edge_set):
    np_points = np.array(points)
    x = np_points[:,0]
    y = np_points[:,1]
    plt.scatter(x, y)

    for row in rows:
        for i in range(len(row)-1):
            j = i + 1
            if type(row[i]) is not tuple or type(row[j]) is not tuple:
                continue
            edge = (row[i][0], row[j][0])
            p1, p2 = row[i][1], row[j][1]

            x1, y1= p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            if edge in true_edge_set:
                plt.plot([x1,x2], [y1,y2], color='y')
            else:
                plt.plot([x1,x2], [y1,y2], color='r')

    plt.show()



