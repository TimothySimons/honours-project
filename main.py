import numpy as np

import geo_shape
from ant_colony import AntColony

points1 = geo_shape.geojson_to_points('resources/1/point_detections.geojson')
ant_colony1 = AntColony(np.array(points1))
ant_colony1.main_loop(50, 10)

ant_colony1.plot_pheromones()

# points2 = geo_shape.geojson_to_points('resources/2/point_detections.geojson')
# ant_colony2 = AntColony(np.array(points2), 50)
# ant_colony2.find_rows()
# ant_colony2.plot()

points3 = geo_shape.geojson_to_points('resources/3/point_detections.geojson')
ant_colony3 = AntColony(np.array(points3))
ant_colony3.main_loop(50, 10)
ant_colony3.plot_pheromones()
