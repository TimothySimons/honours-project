import geo_shape
from ant_colony import AntColony

points = geo_shape.geojson_to_points('resources/1/point_detections.geojson')
ant_colony = AntColony(points, 15)
ant_colony.find_rows()
ant_colony.plot()
