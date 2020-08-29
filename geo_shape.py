import geojson
from geojson import Point, MultiLineString, Feature, FeatureCollection

crs = { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::22293" }}

def geojson_to_points(file_path):
    points = [] 
    with open(file_path) as file:
        geojson_data = geojson.load(file)
        for feature in geojson_data.features:
            points.append(feature.geometry.coordinates)
    return points


def points_to_geojson(file_path, points):
    features = []
    for point in points:
        point = Point(point)
        point_feat = Feature(geometry=point)
        features.append(point_feat)
    feature_collection = FeatureCollection(features, crs=crs)
    with open(file_path, 'w') as f:
        geojson.dump(feature_collection, f)


def rows_to_geojson(file_path, rows):
    features = []
    for row in rows:
        multi_line_string = MultiLineString([row])
        row_feat = Feature(geometry=multi_line_string)
        features.append(row_feat)
    feature_collection = FeatureCollection(features, crs=crs)
    with open(file_path, 'w') as f:
        geojson.dump(feature_collection, f)


def geojson_to_rows(file_path):
    rows = []
    with open(file_path) as file:
        geojson_data = geojson.load(file)
        for feature in geojson_data.features:
            rows.append(feature.geometry.coordinates[0])
    return rows


if __name__ == '__main__':
    points = geojson_to_points('resources/1/point_detections.geojson')
