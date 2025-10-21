"""Rebuild averaged skeletons from raw runs in trails_db/raw.
Groups nearby raw runs by centroid (DBSCAN with eps=50m) and runs skeletonize_runs on each group.
Saves results to trails_db/way/<group_base>_merged_skeleton.json
"""
import os
import json
import math
from sklearn.cluster import DBSCAN
import importlib.util
import sys

THIS_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(THIS_DIR, 'trails_db', 'raw')
WAY_DIR = os.path.join(THIS_DIR, 'trails_db', 'way')

# import route_save module from same folder
spec = importlib.util.spec_from_file_location('route_save', os.path.join(THIS_DIR, 'route_save.py'))
route_save = importlib.util.module_from_spec(spec) # things its non existant :(
spec.loader.exec_module(route_save)  # type: ignore

# collect raw files
raw_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith('_raw.json')]
if not raw_files:
    print('No raw files found in', RAW_DIR)
    sys.exit(0)

centroids = []
raw_runs = []
for path in raw_files:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print('failed to load', path, e)
        continue
    pts = d.get('latlng', {}).get('data', [])
    if not pts:
        continue
    # centroid
    mean_lat = sum(p[0] for p in pts) / len(pts)
    mean_lng = sum(p[1] for p in pts) / len(pts)
    centroids.append((mean_lat, mean_lng))
    #  run with elevations if available
    alts = d.get('altitude', {}).get('data', []) if 'altitude' in d else []
    run = []
    if alts and len(alts) >= len(pts):
        for i in range(min(len(pts), len(alts))):
            run.append([pts[i][0], pts[i][1], alts[i]])
    else:
        for p in pts:
            run.append([p[0], p[1]])
    raw_runs.append({'path': path, 'base': os.path.splitext(os.path.basename(path))[0], 'run': run})

# project cntroid to meters using route_save helper
to_meters, to_latlng, units = route_save._get_projection_funcs(centroids)
coords_m = [to_meters(lat, lng) for lat, lng in centroids]

# cluster centroids
XY = [[x, y] for x, y in coords_m]
cl = DBSCAN(eps=50.0, min_samples=1).fit(XY) # XY cant be assigned according to vscode?
labels = cl.labels_

groups = {}
for i, lbl in enumerate(labels):
    groups.setdefault(int(lbl), []).append(raw_runs[i])

print(f'Found {len(groups)} groups for {len(raw_runs)} raw runs')

os.makedirs(WAY_DIR, exist_ok=True)
for lbl, items in groups.items():
    print(f'Group {lbl}: {len(items)} runs')
    runs = [it['run'] for it in items]
    try:
        merged_list = route_save.skeletonize_runs(runs, spacing_m=5.0, window_m=5.0)
        if not merged_list:
            print('  no merged points for group', lbl)
            continue
        base = items[0]['base']
        # merged_list may contain multiple skeletons (mainline + branches)
        for idx_sk, merged in enumerate(merged_list):
            out_id = f"{base}_merged_skeleton"
            if len(merged_list) > 1:
                out_id = f"{base}_merged_skeleton_branch{idx_sk}"
            latlngs = [[p[0], p[1]] for p in merged]
            elevs = [p[2] for p in merged if len(p) > 2]
            streams = {"latlng": {"data": latlngs, "series_type": "distance", "original_size": len(latlngs), "resolution": "skeleton"}}
            route_save.save_trail_streams(out_id, streams, folder=WAY_DIR, elevation_samples=elevs)
            print('  saved', out_id, 'points:', len(latlngs))
    except Exception as e:
        print('  failed to merge group', lbl, e)

print('Done')
