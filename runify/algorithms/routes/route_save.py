import json
import os
from typing import Dict
import geopandas as gpd
from shapely.geometry import Point

def save_trail_streams(trail_id: str, new_streams: Dict, folder: str = 'trails_db'):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'{trail_id}.json')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            trail_data = json.load(f)
    else:
        trail_data = {k: {'data': [], 'series_type': '', 'original_size': 0, 'resolution': ''} for k in new_streams}
    for key in new_streams:
        stream = new_streams[key]
        if key not in trail_data:
            trail_data[key] = stream
        else:
            trail_data[key]['data'] += stream['data']
            trail_data[key]['original_size'] += len(stream['data'])
            trail_data[key]['series_type'] = stream.get('series_type', trail_data[key]['series_type'])
            trail_data[key]['resolution'] = stream.get('resolution', trail_data[key]['resolution'])
    with open(path, 'w') as f:
        json.dump(trail_data, f)
    print(f"Trail data for '{trail_id}' updated and saved at {path} (points: {new_streams['latlng']['original_size']})")
    return f"Trail data for {trail_id} updated and saved at {path}"

def assign_points_to_trails(gps_coords, osm_trails_gdf, threshold=25):
    segments = []
    current_trail_id, current_points = None, []
    for idx, (lat, lng) in enumerate(gps_coords):
        pt = Point(lng, lat)
        min_dist, best_trail_id = float('inf'), None
        for _, row in osm_trails_gdf.iterrows():
            trail_line = row.geometry
            candidate_id = str(row['id'])  # <-- CHANGED HERE!
            dist = pt.distance(trail_line) * 111139 # degrees to meters
            if dist < min_dist:
                min_dist = dist
                best_trail_id = candidate_id
        if min_dist <= threshold:
            if current_trail_id != best_trail_id:
                if current_trail_id and current_points:
                    segments.append((current_trail_id, current_points))
                current_trail_id = best_trail_id
                current_points = []
                print(f"--- New segment started for '{best_trail_id}' ---")
            current_points.append([lat, lng])
        else:
            if current_trail_id and current_points:
                segments.append((current_trail_id, current_points))
            current_trail_id, current_points = None, []
    if current_trail_id and current_points:
        segments.append((current_trail_id, current_points))
    print(f"Total segments found: {len(segments)}")
    return segments


# ---- MAIN OPERATIONS ----

file = 'stream_15914708833.json'
with open(file, 'r') as f:
    example_streams = json.load(f)

gps_coords = example_streams['latlng']['data'] 

osm_trails = gpd.read_file("osm_trails_large.geojson")

print("OSM columns:", osm_trails.columns)
print(osm_trails.head())

segments = assign_points_to_trails(gps_coords, osm_trails, threshold=25)

for idx, (trail_id, points_segment) in enumerate(segments):
    row = osm_trails[osm_trails['id'] == trail_id]   # <-- CHANGED HERE!
    if len(row) == 0:
        print(f"-- Warning: OSM trail {trail_id} not found, skipping.")
        continue
    streams_segment = {
        "latlng": {
            "data": points_segment,
            "series_type": "distance",
            "original_size": len(points_segment),
            "resolution": "high"
        }
    }
    print(f">>> Saving segment {idx+1}/{len(segments)} for trail '{trail_id}' with {len(points_segment)} points...")
    save_trail_streams(trail_id, streams_segment)

