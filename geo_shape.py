import geojson

def geojson_to_points(file_path):
    points = [] 
    with open(file_path) as file:
        geojson_data = geojson.load(file)
        for feature in geojson_data.features:
            points.append(feature.geometry.coordinates)
    return points

if __name__ == '__main__':
    points = geojson_to_points('resources/1/point_detections.geojson')
