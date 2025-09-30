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
            trail_line = row.geometry
            candidate_id = str(row['id'])
            dist = pt.distance(trail_line) * 111139  # degrees to meters
            if dist < min_dist:
                min_dist = dist
                best_trail_id = candidate_id
                best_trail_line = trail_line
        if min_dist <= threshold and best_trail_line is not None:
            if current_trail_id != best_trail_id:
                if current_trail_id and current_points:
                    segments.append((current_trail_id, current_points))
                current_trail_id = best_trail_id
                current_points = []
                print(f"--- New segment started for '{best_trail_id}' ---")
            snap_point = best_trail_line.interpolate(best_trail_line.project(pt))
            snap_lat, snap_lng = snap_point.y, snap_point.x
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

# ---- MAIN OPERATIONS ----

file = 'stream_15876634239.json'
with open(file, 'r') as f:
    example_streams = json.load(f)

latlngs = example_streams['latlng']['data']
altitudes = example_streams['altitude']['data']

gps_coords = [latlngs[i] + [altitudes[i]] for i in range(min(len(latlngs), len(altitudes)))]

osm_trails = gpd.read_file("trails_db/bbox_raw_saves/export.geojson")

print("OSM columns:", osm_trails.columns)
print(osm_trails.head())

segments = assign_points_to_trails(gps_coords, osm_trails, threshold=25)

if not segments or all(len(seg[1]) == 0 for seg in segments):
    print(f"No matched OSM segments—saving entire uploaded route as custom.")
    # Save whole GPS track as its own trail (or use a prefix/postfix like "_custom")
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
    trail_id = file.replace('.json', '') + "_custom"
    save_trail_streams(trail_id, streams_segment, elevation_samples=elevation_samples)
    print(f"Custom route '{trail_id}' saved with {len(latlng_list)} points.")
    avg_point = average_trail_points(gps_coords)
    if avg_point is not None:
        avg_streams = {
            "latlng": {
                "data": [avg_point[:2]],  # whole route mean
                "series_type": "distance",
                "original_size": 1,
                "resolution": "canonical"
            },
            "elevation_stats": {
                "samples": [avg_point[2]],
                "mean": avg_point[2],
                "count": 1
            }
        }
        save_trail_streams(trail_id + "_canonical", avg_streams, elevation_samples=[avg_point[2]])
        print(f"Canonical OSM-independent route for '{trail_id}': {avg_point}")

for idx, (trail_id, points_segment) in enumerate(segments):
    row = osm_trails[osm_trails['id'] == trail_id]
    if len(row) == 0:
        print(f"-- Warning: OSM trail {trail_id} not found, skipping.")
        continue

    # Raw data for segment
    latlng_list = [[x[0], x[1]] for x in points_segment]
    elevation_samples = [x[2] for x in points_segment if len(x) > 2]
    streams_segment = {
        "latlng": {
            "data": latlng_list,
            "series_type": "distance",
            "original_size": len(latlng_list),
            "resolution": "high"
        }
    }
    print(f">>> Saving segment {idx+1}/{len(segments)} for trail '{trail_id}' with {len(latlng_list)} points...")

    # Output first point for debug
    print("First seg:", points_segment[0])
    print("First elevation:", points_segment[0][2] if len(points_segment[0]) > 2 else "none")

    # 1. Save raw segment points as before
    save_trail_streams(trail_id, streams_segment, elevation_samples=elevation_samples)

    # 2. Save the averaged canonical reference for the segment:
    avg_point = average_trail_points(points_segment)
    if avg_point is not None:
        avg_streams = {
            "latlng": {
                "data": [avg_point[:2]],  # just the avg lat/lng
                "series_type": "distance",
                "original_size": 1,
                "resolution": "canonical"
            },
            "elevation_stats": {
                "samples": [avg_point[2]],
                "mean": avg_point[2],
                "count": 1
            }
        }
        save_trail_streams(trail_id + "_canonical", avg_streams, elevation_samples=[avg_point[2]])
        print(f"Canonical (averaged) trail for '{trail_id}': {avg_point}")

