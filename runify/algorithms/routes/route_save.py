import json
import os
from typing import Dict
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def average_trail_points(points, elev_outlier_thresh=30):
    """
    Given a list of [lat, lng, elev] or [lat, lng], return a single averaged [lat, lng, elev]
    while ignoring elevation outliers.
    """
    # Keep only those points that have elevation, and elevation is sensible
    points_with_elev = [p for p in points if len(p) > 2 and p[2] and p[2] > 0]
    if not points_with_elev:
        return None  # nothing valid
    elevs = np.array([pt[2] for pt in points_with_elev])
    med_elev = np.median(elevs)
    # Remove outliers: points whose elev diff from median > threshold
    filtered_points = [pt for pt in points_with_elev if abs(pt[2] - med_elev) < elev_outlier_thresh]
    arr = np.array(filtered_points)
    # Average lat/lng/elev
    avg_lat = float(np.mean(arr[:, 0]))
    avg_lng = float(np.mean(arr[:, 1]))
    avg_elev = float(np.mean(arr[:, 2]))
    return [avg_lat, avg_lng, avg_elev]

def save_trail_streams(trail_id: str, new_streams: Dict, folder: str = 'trails_db', elevation_samples=None):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'{trail_id}.json')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            trail_data = json.load(f)
    else:
        trail_data = {k: {'data': [], 'series_type': '', 'original_size': 0, 'resolution': ''} for k in new_streams}
    for key in new_streams:
        stream = new_streams[key]
        if isinstance(stream, dict) and 'data' in stream:
            if key not in trail_data:
                trail_data[key] = stream
            else:
                trail_data[key]['data'] += stream['data']
                trail_data[key]['original_size'] += len(stream['data'])
                trail_data[key]['series_type'] = stream.get('series_type', trail_data[key]['series_type'])
                trail_data[key]['resolution'] = stream.get('resolution', trail_data[key]['resolution'])
        else:
            trail_data[key] = stream
    # Elevation enrichment
    if elevation_samples is not None and len(elevation_samples) > 0:
        prev_elev = trail_data.get('elevation_samples', [])
        all_elev = prev_elev + elevation_samples
        elevation_mean = float(np.mean(all_elev)) if all_elev else None
        trail_data['elevation_stats'] = {
            'samples': all_elev,
            'mean': elevation_mean,
            'count': len(all_elev)
        }
        print(f"Elevation stats — mean: {elevation_mean}, count: {len(all_elev)}")
    with open(path, 'w') as f:
        json.dump(trail_data, f)
    print(f"Trail data for '{trail_id}' updated and saved at {path} (points: {new_streams['latlng']['original_size']})")
    return f"Trail data for {trail_id} updated and saved at {path}"

def assign_points_to_trails(gps_coords, osm_trails_gdf, threshold=25):
    segments = []
    current_trail_id, current_points = None, []
    for idx, item in enumerate(gps_coords):
        lat, lng = item[0], item[1]
        elev = item[2] if len(item) > 2 else None
        pt = Point(lng, lat)
        min_dist, best_trail_id = float('inf'), None
        best_trail_line = None
        for _, row in osm_trails_gdf.iterrows():
            geom = row.geometry
            candidate_id = str(row['id'])
            # Ensure we have a linear geometry to work with. If the geometry
            # is a MultiLineString or Polygon, pick the linear component that
            # is closest to the point.
            candidate_line = None
            try:
                gtype = geom.geom_type
            except Exception:
                # malformed geometry, skip
                continue

            if gtype in ('LineString', 'LinearRing'):
                candidate_line = geom
            elif gtype == 'MultiLineString':
                # choose the linestring component nearest to the point
                try:
                    candidate_line = min(geom.geoms, key=lambda g: pt.distance(g))
                except Exception:
                    candidate_line = None
            elif gtype == 'Polygon':
                # use exterior ring as a linear proxy
                try:
                    candidate_line = geom.exterior
                except Exception:
                    candidate_line = None
            elif gtype == 'MultiPolygon':
                # pick the polygon whose exterior is nearest
                try:
                    poly = min(geom.geoms, key=lambda p: pt.distance(p.exterior))
                    candidate_line = poly.exterior
                except Exception:
                    candidate_line = None
            elif gtype == 'GeometryCollection':
                # find any linear components
                try:
                    linear_parts = [g for g in geom.geoms if g.geom_type in ('LineString', 'LinearRing')]
                    if linear_parts:
                        candidate_line = min(linear_parts, key=lambda g: pt.distance(g))
                except Exception:
                    candidate_line = None
            else:
                # unknown geometry type — skip
                candidate_line = None

            if candidate_line is None:
                continue

            try:
                dist = pt.distance(candidate_line) * 111139  # degrees to meters
            except Exception:
                # if distance computation fails for this candidate, skip
                continue

            if dist < min_dist:
                min_dist = dist
                best_trail_id = candidate_id
                best_trail_line = candidate_line
        if min_dist <= threshold and best_trail_line is not None:
            if current_trail_id != best_trail_id:
                if current_trail_id and current_points:
                    segments.append((current_trail_id, current_points))
                current_trail_id = best_trail_id
                current_points = []
                print(f"--- New segment started for '{best_trail_id}' ---")
            # project + interpolate can fail if geometry is degenerate; guard it
            try:
                proj = best_trail_line.project(pt)
                snap_point = best_trail_line.interpolate(proj)
                snap_lat, snap_lng = snap_point.y, snap_point.x
            except Exception:
                # can't snap to this geometry; skip this point
                continue
            if elev is not None:
                current_points.append([snap_lat, snap_lng, elev])
            else:
                current_points.append([snap_lat, snap_lng])
        else:
            if current_trail_id and current_points:
                segments.append((current_trail_id, current_points))
            current_trail_id, current_points = None, []
    if current_trail_id and current_points:
        segments.append((current_trail_id, current_points))
    print(f"Total segments found: {len(segments)}")
    return segments

def process_stream_file(file: str, trails_folder: str = 'trails_db'):
    """Process a single stream JSON file and save segments/canonical points.

    This function mirrors the behaviour of the original script but is
    reusable: pass the path to a stream JSON (relative or absolute) and
    an optional trails_folder where `bbox_raw_saves/export.geojson` and
    `way` folders exist.
    """
    with open(file, 'r') as f:
        example_streams = json.load(f)

    latlngs = example_streams.get('latlng', {}).get('data', [])
    altitudes = example_streams.get('altitude', {}).get('data', [])

    gps_coords = []
    if altitudes and len(altitudes) >= len(latlngs):
        gps_coords = [latlngs[i] + [altitudes[i]] for i in range(min(len(latlngs), len(altitudes)))]
    else:
        gps_coords = [p for p in latlngs]

    bbox_geojson = os.path.join(trails_folder, 'bbox_raw_saves', 'export.geojson')
    if not os.path.isfile(bbox_geojson):
        print(f"OSM bbox file not found at {bbox_geojson}. Will save route as custom.")
        latlng_list = [[x[0], x[1]] for x in gps_coords]
        elevation_samples = [x[2] for x in gps_coords if len(x) > 2]
        streams_segment = {
            "latlng": {
                "data": latlng_list,
                "series_type": "distance",
                "original_size": len(latlng_list),
                "resolution": "high"
            }
        }
        trail_id = os.path.splitext(os.path.basename(file))[0] + "_custom"
        save_trail_streams(trail_id, streams_segment, folder=trails_folder, elevation_samples=elevation_samples)
        avg_point = average_trail_points(gps_coords)
        if avg_point is not None:
            avg_streams = {
                "latlng": {"data": [avg_point[:2]], "series_type": "distance", "original_size": 1, "resolution": "canonical"},
                "elevation_stats": {"samples": [avg_point[2]], "mean": avg_point[2], "count": 1}
            }
            save_trail_streams(trail_id + "_canonical", avg_streams, folder=trails_folder, elevation_samples=[avg_point[2]])
        return

    import geopandas as gpd

    osm_trails = gpd.read_file(bbox_geojson)
    print("OSM columns:", osm_trails.columns)
    segments = assign_points_to_trails(gps_coords, osm_trails, threshold=25)

    if not segments or all(len(seg[1]) == 0 for seg in segments):
        print(f"No matched OSM segments—saving entire uploaded route as custom.")
        latlng_list = [[x[0], x[1]] for x in gps_coords]
        elevation_samples = [x[2] for x in gps_coords if len(x) > 2]
        streams_segment = {
            "latlng": {
                "data": latlng_list,
                "series_type": "distance",
                "original_size": len(latlng_list),
                "resolution": "high"
            }
        }
        trail_id = os.path.splitext(os.path.basename(file))[0] + "_custom"
        save_trail_streams(trail_id, streams_segment, folder=trails_folder, elevation_samples=elevation_samples)
        avg_point = average_trail_points(gps_coords)
        if avg_point is not None:
            avg_streams = {
                "latlng": {"data": [avg_point[:2]], "series_type": "distance", "original_size": 1, "resolution": "canonical"},
                "elevation_stats": {"samples": [avg_point[2]], "mean": avg_point[2], "count": 1}
            }
            save_trail_streams(trail_id + "_canonical", avg_streams, folder=trails_folder, elevation_samples=[avg_point[2]])
        return

    for idx, (trail_id, points_segment) in enumerate(segments):
        row = osm_trails[osm_trails['id'] == trail_id] if 'id' in osm_trails.columns else osm_trails[osm_trails['id'] == int(trail_id)]
        if len(row) == 0:
            print(f"-- Warning: OSM trail {trail_id} not found, skipping.")
            continue

        latlng_list = [[x[0], x[1]] for x in points_segment]
        elevation_samples = [x[2] for x in points_segment if len(x) > 2]
        streams_segment = {
            "latlng": {"data": latlng_list, "series_type": "distance", "original_size": len(latlng_list), "resolution": "high"}
        }
        print(f">>> Saving segment {idx+1}/{len(segments)} for trail '{trail_id}' with {len(latlng_list)} points...")
        save_trail_streams(trail_id, streams_segment, folder=trails_folder, elevation_samples=elevation_samples)

        avg_point = average_trail_points(points_segment)
        if avg_point is not None:
            avg_streams = {
                "latlng": {"data": [avg_point[:2]], "series_type": "distance", "original_size": 1, "resolution": "canonical"},
                "elevation_stats": {"samples": [avg_point[2]], "mean": avg_point[2], "count": 1}
            }
            save_trail_streams(trail_id + "_canonical", avg_streams, folder=trails_folder, elevation_samples=[avg_point[2]])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-stream', help='Path to a stream JSON file to process and save into trails_db')
    parser.add_argument('--trails-folder', default='trails_db', help='Folder where trails_db is located')
    args = parser.parse_args()
    if args.process_stream:
        process_stream_file(args.process_stream, trails_folder=args.trails_folder)

