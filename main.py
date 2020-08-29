import numpy as np
import yaml

import analysis
import geo_shape
from ant_colony import AntColony



with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)


#points1 = geo_shape.geojson_to_points('resources/1/point_detections.geojson')
#ant_colony1 = AntColony(np.array(points1))
#final_rows = ant_colony1.main_loop(10, 10, 10)
#analysis.plot_rows(points1, final_rows)

#geo_shape.rows_to_geojson('../../rows.geojson', final_rows)


#points2 = geo_shape.geojson_to_points('resources/2/point_detections.geojson')
#ant_colony2 = AntColony(np.array(points2))
#final_rows = ant_colony2.main_loop(10, 10, 10)
#analysis.plot_rows(points2, final_rows)

points3 = geo_shape.geojson_to_points('resources/3/point_detections.geojson')
ant_colony3 = AntColony(np.array(points3), config)
final_rows = ant_colony3.main_loop(15, 5, 15)
analysis.plot_rows(points3, final_rows)

#geo_shape.points_to_geojson('../../something.geojson', points3)

#import cProfile
#cProfile.run('ant_colony1.main_loop(10, 10, 10)', 'ACO_stats')

#import pstats
#from pstats import SortKey
#p = pstats.Stats('ACO_stats')
#p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()



