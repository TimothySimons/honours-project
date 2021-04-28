"""Module for reading and writing geojson files of line and point shapes.

The CRS mapping used in this project is that of urn:ogc:def:crs:EPSG::22293
corresponding to the location of all considered inputs for this project.
"""


import geojson
from geojson import Point, MultiLineString, Feature, FeatureCollection


def geojson_to_points(file_path):
    """Converts a geojson file of Point shapes to a list of points."""
    points = []
    with open(file_path) as f:
        geojson_data = geojson.load(f)
        for feature in geojson_data.features:
            points.append(feature.geometry.coordinates)
    return points


def points_to_geojson(file_path, points, crs):
    """Converts a list of points to a geojson file of Point shapes."""
    features = []
    for point in points:
        point = Point(point)
        point_feat = Feature(geometry=point)
        features.append(point_feat)
    feature_collection = FeatureCollection(features, crs=crs)
    with open(file_path, 'w') as f:
        geojson.dump(feature_collection, f)


def geojson_to_rows(file_path):
    """Converts a geojson of MultiLineString shapes to a list of rows."""
    rows = []
    with open(file_path) as f:
        geojson_data = geojson.load(f)
        for feature in geojson_data.features:
            rows.append(feature.geometry.coordinates[0])
    return rows


def rows_to_geojson(file_path, rows, crs):
    """Converts a list of rows to a geojson file of MultiLineString shapes."""
    features = []
    for row in rows:
        multi_line_string = MultiLineString([row])
        row_feat = Feature(geometry=multi_line_string)
        features.append(row_feat)
    feature_collection = FeatureCollection(features, crs=crs)
    with open(file_path, 'w') as f:
        geojson.dump(feature_collection, f)
