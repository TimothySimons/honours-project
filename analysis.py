import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import yaml
from matplotlib import rcParams

import geo_shape
from ant import Ant
from ant_colony import AntColony


def construct_ground_truth(row_points):
    """Constructs ground truth row from hand labelled row set."""
    kd_tree = scipy.spatial.KDTree(row_points)
    row_i = [random.randrange(0,len(row_points) - 1)]
    construct_partial(row_i, row_points, kd_tree)
    row_i.reverse()
    construct_partial(row_i, row_points, kd_tree)
    row = [row_points[i] for i in row_i]
    return row


def construct_partial(row_i, row_points, kd_tree, k=10):
    """Constructs partial row in the direction of nearest next point."""
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
    """Checks that the next point keeps the row going in the forward direction."""
    row = [row_points[i] for i in row_i]
    p2 = row_points[next_i]
    if len(row) > 1:
        p0 = row[-2]
        p1 = row[-1]
        angle = get_abs_angle(p0, p1, p2)
        if angle < 90:
            return False
    return True


def get_abs_angle(p0, p1, p2):
    """Returns the angle at vertex p1 enclosed by rays p0 p1 and p1 p2."""
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return abs(np.degrees(angle))


def construct_edge_set(model_rows, true_rows):
    """Constructus edge sets for both model produced rows and true rows."""
    current_index = 0
    for m_row in model_rows:
        for t_row in true_rows:
            current_index = assign_indices(m_row, t_row, current_index)
    m_edge_set = rows_to_edge_set(model_rows)
    t_edge_set = rows_to_edge_set(true_rows)
    return m_edge_set, t_edge_set


def assign_indices(model_row, true_row, current_index):
    """Assigns common index if the same point exists in model_row and true_row."""
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
    """Converts a series of rows into an edge set.

    Rows consisting of only a single point do not form part of the edge set.
    """
    edge_set = []
    for row in rows:
        for i in range(len(row) - 1):
            j = i + 1
            if type(row[i]) is not tuple or type(row[j]) is not tuple:
                edge_set.append((None, None))
            else:
                point_i = row[i][0]
                point_j = row[j][0]
                edge_set.append(tuple(sorted([point_i, point_j])))
    return edge_set


def precision(model_edge_set, true_edge_set):
    """Calculates the precision of model rows when compared to true rows."""
    true_positive = 0
    false_positive = 0
    for edge in model_edge_set:
        if edge in true_edge_set and edge is not (None, None):
            true_positive += 1
        else:
            false_positive += 1
    return true_positive / (true_positive + false_positive)
    


def recall(model_edge_set, true_edge_set):
    """Calculates the recall of model rows when compared to true rows."""
    true_positive = 0
    for edge in model_edge_set:
        if edge in true_edge_set and edge is not (None, None):
            true_positive += 1

    false_negative = 0
    for edge in true_edge_set:
        if edge not in model_edge_set or edge is (None, None):
            false_negative += 1
    return true_positive / (true_positive + false_negative)


def iou(model_edge_set, true_edge_set):
    """Calculates the iou of model rows when compared to true rows."""
    true_positive = 0
    false_positive = 0
    for edge in model_edge_set:
        if edge in true_edge_set and edge is not (None, None):
            true_positive += 1
        else:
            false_positive += 1

    false_negative = 0
    for edge in true_edge_set:
        if edge not in model_edge_set or edge is (None, None):
            false_negative += 1

    return true_positive / (true_positive + false_positive + false_negative)


def plot_rows(points, rows):
    """Plots lines and points that represent rows and trees respectively."""
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


def plot_trails(del_points, points, pheromone_matrix):
    np_points = np.array(del_points)
    x = np_points[:,0]
    y = np_points[:,1]
    plt.scatter(x, y, color='gray')

    pher_values = []
    for (index_1, index_2) , pher_value in pheromone_matrix.items():
        pher_values.append(pher_value)
        edge_x  =  [points[index_1][0], points[index_2][0]]
        edge_y  =  [points[index_1][1], points[index_2][1]]
        plt.plot(edge_x, edge_y, linewidth=pher_value, color='r')

    plt.show()    

    plt.plot(pher_values)
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


def plot_results(precision, recall, iou):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.set(ylim=(0.8,1), xlim=(-0.05,1.05))
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    ax.plot(precision[:,0], precision[:,1], label='precision', color='tab:green')
    ax.plot(recall[:,0], recall[:,1], label='recall', color='tab:blue')
    ax.plot(iou[:,0], iou[:,1], label='IoU', color='tab:red')
    plt.show()

def metric_rows(model_edge_set, true_edge_set, model_rows, true_rows):
    mr_dict = {}
    for row in model_rows:
        r_dict = {p[0]:p[1] for p in row}
        mr_dict.update(r_dict)

    tr_dict = {}
    for row in true_rows:
        r_dict = {p[0]:p[1] for p in row}
        tr_dict.update(r_dict)

    tp_edges = []
    fp_edges = []
    fn_edges = []
    for edge in model_edge_set:
        i, j = edge
        p1 = mr_dict[i]
        p2 = mr_dict[j]
        r = [p1, p2] 
        if edge in true_edge_set and edge is not (None, None):
            tp_edges.append(r)
        else:
            fp_edges.append(r)


    for edge in true_edge_set:
        i, j = edge
        if edge not in model_edge_set or edge is (None, None):
            p1 = tr_dict[i]
            p2 = tr_dict[j]
            r = [p1, p2]
            fn_edges.append(r)

    geo_shape.rows_to_geojson('resources/1/tp.geojson', tp_edges)
    geo_shape.rows_to_geojson('resources/1/fp.geojson', fp_edges)
    geo_shape.rows_to_geojson('resources/1/fn.geojson', fn_edges)


def experiment(config, points):
    file_path = 'resources/1/weights_2.txt'
    f = open(file_path, 'w')
    for i in np.arange(0, 1.05, 0.05):
        config['ant']['dist_weight'] = i
        config['ant']['angle_weight'] = 1 - i
        ant_colony = AntColony(np.array(points), config)
        model_rows = ant_colony.main_loop(15, 5, 15)
        true_rows = geo_shape.geojson_to_rows('resources/1/true_rows.geojson')
        model_edge_set, true_edge_set = construct_edge_set(model_rows, true_rows)
        precision_stat = precision(model_edge_set, true_edge_set)
        recall_stat = recall(model_edge_set, true_edge_set)
        iou_stat = iou(model_edge_set, true_edge_set)
        print(iou_stat)
        import sys; sys.stdout.flush()
        f.write(str(i) + ' ' + str(precision_stat) + '\n')
        f.write(str(i) + ' ' + str(recall_stat) + '\n')
        f.write(str(i) + ' ' + str(iou_stat) + '\n')
    f.close()


def visualise(file_path):
    f = open(file_path)
    precision = []
    recall = []
    iou = []
    while True:
        line = f.readline()
        line2 = f.readline()
        line3 = f.readline()
        if not line:
            break
        p_pair = list(map(float, line.split(' ')))
        r_pair = list(map(float, line2.split(' ')))
        i_pair = list(map(float, line3.split(' ')))
        precision.append(p_pair)
        recall.append(r_pair)
        iou.append(i_pair)
    return precision, recall, iou


        
        

#if __name__ == '__main__':
#   precision_1, recall_1, iou_1 = visualise('../results/1/weights_1.txt')
   #precision_2, recall_2, iou_2 = visualise('../results/2/num_iter_2.txt')
   #precision_3, recall_3, iou_3 = visualise('../results/2/num_iter_3.txt')
#   avg = lambda s1, s2, s3: (s1[0], (s1[1] + s2[1] + s3[1]) / 3)

   #precision = list(map(avg, precision_1, precision_2, precision_3))
   #recall = list(map(avg, recall_1, recall_2, recall_3))
   #iou = list(map(avg, iou_1, iou_2, iou_3))
#   plot_results(np.array(precision_1), np.array(recall_1), np.array(iou_1))

        


if __name__ == '__main__':
   with open("config.yml", 'r') as stream: config = yaml.safe_load(stream)

   points = geo_shape.geojson_to_points('resources/1/point_detections.geojson')
   experiment(config, points)


#if __name__ == '__main__':
#
#    with open("config.yml", 'r') as stream:
#        config = yaml.safe_load(stream)
#
#    points = geo_shape.geojson_to_points('resources/1/point_detections.geojson')
#    ant_colony = AntColony(np.array(points), config)
#
#    model_rows = ant_colony.main_loop(1, 5, 1)
#    geo_shape.rows_to_geojson('resources/1/model_rows.geojson', model_rows)
#    true_rows = geo_shape.geojson_to_rows('resources/1/true_rows.geojson')
#    model_edge_set, true_edge_set = construct_edge_set(model_rows, true_rows)
#    precision_stat = precision(model_edge_set, true_edge_set)
#    recall_stat = recall(model_edge_set, true_edge_set)
#    iou_stat = iou(model_edge_set, true_edge_set)
#
#    metric_rows(model_edge_set, true_edge_set, model_rows, true_rows)
#    print('detection samples: ', len(points))
#    print('precision: ', precision_stat)
#    print('recall: ', recall_stat)
#    print('iou: ', iou_stat)
