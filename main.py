import numpy as np
import yaml

import analysis
import geo_shape
from ant_colony import AntColony



with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

points = geo_shape.geojson_to_points('resources/3/point_detections.geojson')
ant_colony = AntColony(np.array(points), config)
final_rows = ant_colony.main_loop(1, 5, 1)




#points = geo_shape.geojson_to_points('resources/1/point_detections.geojson')
#precision = []
#recall = []
#for num_neigh in range(1,35, 10):
#    true_rows = geo_shape.geojson_to_rows('resources/1/true_rows.geojson')
#    ant_colony = AntColony(np.array(points), config)
#    final_rows = ant_colony.main_loop(40, 5, 15)
#    m_edge_set, t_edge_set = analysis.construct_edge_set(final_rows, true_rows)
#    precision.append(analysis.precision(m_edge_set, t_edge_set))
#    recall.append(analysis.recall(m_edge_set, t_edge_set))
#    print('neighbours', num_neigh)
#    print('precision: ', precision)
#    print('recall:', recall)
#    print()
#    import sys
#    sys.stdout.flush()
#    break


#import matplotlib.pyplot as plt
#plt.plot(precision)
#plt.show()
#plt.plot(recall)
#plt.show()
#for num_iter in range(1,27,5):
#    true_rows = geo_shape.geojson_to_rows('resources/1/true_rows.geojson')
#    ant_colony = AntColony(np.array(points), config)
#    final_rows = ant_colony.main_loop(25, 5, 25)
    #analysis.plot_rows(points, final_rows)
#    m_edge_set, t_edge_set = analysis.construct_edge_set(final_rows, true_rows)
#    precision.append(analysis.precision(m_edge_set, t_edge_set))
#    recall.append(analysis.recall(m_edge_set, t_edge_set))
#    print('number of iterations: ', num_iter)
#    print('precision: ', precision)
#    print('recall:', recall)
#    import sys
#    sys.stdout.flush()

