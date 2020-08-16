import numpy as np

import analysis
import geo_shape
from ant_colony import AntColony

points1 = geo_shape.geojson_to_points('resources/1/point_detections.geojson')
ant_colony1 = AntColony(np.array(points1))
ant_colony1.main_loop(10, 5, 1)
analysis.plot(points1, ant_colony1.candidate_rows, ant_colony1.end_points)

# points2 = geo_shape.geojson_to_points('resources/2/point_detections.geojson')
# ant_colony2 = AntColony(np.array(points2), 50)
# ant_colony2.find_rows()
# ant_colony2.plot()

#points3 = geo_shape.geojson_to_points('resources/3/point_detections.geojson')
#ant_colony3 = AntColony(np.array(points3))
#ant_colony3.main_loop(50, 10, 10)
#analysis.plot(points3, ant_colony3.candidate_rows, ant_colony3.end_points)
