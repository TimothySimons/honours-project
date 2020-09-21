import matplotlib.pyplot as plt
import numpy as np
import yaml

import analysis
import geo_shape
from ant_colony import AntColony


def plot(points, orig_points):
    np_points = np.array(points)
    x = np_points[:,0]
    y = np_points[:,1]
    plt.scatter(x, y)

    np_points = np.array(orig_points)
    x = np_points[:,0]
    y = np_points[:,1]
    plt.scatter(x, y)
    
    plt.show()   


def get_row_groups(file_path):
    points = []
    row_groups = []
    for i in range(1,139):
        file_name = 'C:/Users/smnti/Desktop/annotated_rows_3/row_{}.txt'.format(i)
        f = open(file_name, 'r')
        lines = f.readlines()
        row_group = []
        for line in lines:
            line = line.rstrip('\n')
            line_split = line.split()
            x, y = float(line_split[0]), float(line_split[1])
            points.append([x, y])
            row_group.append([x, y])
        row_groups.append(row_group)
    return row_groups


with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

file_path = 'C:/Users/smnti/Desktop/annotated_rows_3/row_{}.txt'.format(i)
row_groups = get_row_groups(file_path)
rows = []
points = []
for row_points in row_groups:
    if row_points == []: break
    row = analysis.construct_ground_truth(row_points)
    rows.append(row)
    points = points + row

geo_shape.rows_to_geojson('../../true_rows.geojson', rows)






