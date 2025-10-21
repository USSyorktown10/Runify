import json
import os
from typing import Dict
import numpy as np
import math
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def average_centerline(points, merge_thresh=10, elev_outlier_thresh=30):
    """
    Cluster points by proximity (merge_thresh in meters),
    then average each cluster: [lat, lng, elev].
    Returns: list of averaged [lat, lng, elev]
    """
    # Build array of lat/lng and project roughly to meters for local clustering
    coords = np.array([[p[0], p[1]] for p in points])
    mean_lat = float(np.mean(coords[:, 0]))
    meters = np.zeros_like(coords)
    meters[:, 0] = coords[:, 0] * 111139
    meters[:, 1] = coords[:, 1] * 111139 * np.cos(np.radians(mean_lat))

    # Cluster nearby points into groups (eps in meters)
    clustering = DBSCAN(eps=merge_thresh, min_samples=1).fit(meters)
    labels = clustering.labels_
    unique_labels = sorted(set(labels))

    avg_points = []
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            continue
        clust = [points[i] for i in idxs]
        lats = [p[0] for p in clust]
        lngs = [p[1] for p in clust]
        avg_lat = float(np.mean(lats))
        avg_lng = float(np.mean(lngs))
        # Average elevation removing outliers by median
        elevs = [p[2] for p in clust if len(p) > 2 and p[2] is not None]
        avg_pt = [avg_lat, avg_lng]
        if elevs:
            med = float(np.median(elevs))
            elevs_f = [e for e in elevs if abs(e - med) < elev_outlier_thresh]
            if elevs_f:
                avg_elev = float(np.mean(elevs_f))
                avg_pt.append(avg_elev)
        avg_points.append(avg_pt)
    return avg_points

def save_trail_streams(trail_id: str, new_streams: Dict, folder: str = 'trails_db/way', elevation_samples=None):
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
                before = len(trail_data[key]['data'])
                added = len(stream['data'])
                trail_data[key]['data'] += stream['data']
                after = len(trail_data[key]['data'])
                print(f"[DEBUG] save_trail_streams merged {added} points into '{key}': {before} -> {after}")
                trail_data[key]['original_size'] += added
                trail_data[key]['series_type'] = stream.get('series_type', trail_data[key]['series_type'])
                trail_data[key]['resolution'] = stream.get('resolution', trail_data[key]['resolution'])
        else:
            trail_data[key] = stream
    # Elevation average and stats
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


def _haversine_meters(a_lat, a_lng, b_lat, b_lng):
    # returns distance in meters between two lat/lng points
    R = 6371000.0
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    dlat = lat2 - lat1
    dlng = math.radians(b_lng - a_lng)
    x = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2.0)**2
    return 2 * R * math.asin(math.sqrt(x))


def merge_centerline_into_trail(trail_id: str, points_segment, folder: str = 'trails_db/way', snap_threshold_m=15):
    """Merge an incoming points_segment into the existing trail file (per-point averaging).

    The trail JSON will keep a 'latlng' list under which coordinates are updated,
    and a parallel 'latlng_meta' list containing per-point elevation samples and counts.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{trail_id}.json")
    if os.path.isfile(path):
        try:
            with open(path, 'r') as f:
                trail_data = json.load(f)
        except Exception:
            trail_data = {}
    else:
        trail_data = {}

    existing_pts = trail_data.get('latlng', {}).get('data', [])
    meta = trail_data.get('latlng_meta', [])

    # make sure meta aligns with existing_pts
    while len(meta) < len(existing_pts):
        meta.append({'samples': [], 'count': 0, 'mean_elev': None})

    # normalize incoming points
    incoming = []
    for p in points_segment:
        if not p or len(p) < 2:
            continue
        lat = float(p[0]); lng = float(p[1])
        elev = float(p[2]) if len(p) > 2 else None
        incoming.append((lat, lng, elev))

    for lat, lng, elev in incoming:
        best_idx = None
        best_dist = float('inf')
        for i, ex in enumerate(existing_pts):
            ex_lat, ex_lng = float(ex[0]), float(ex[1])
            d = _haversine_meters(lat, lng, ex_lat, ex_lng)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None and best_dist <= snap_threshold_m:
            # update existing point by running average
            n = meta[best_idx].get('count', 0)
            if n <= 0:
                # initialize
                meta[best_idx]['count'] = 1
                meta[best_idx]['samples'] = []
                meta[best_idx]['mean_elev'] = None
                n = 1
            # compute new average lat/lng
            ex_lat, ex_lng = float(existing_pts[best_idx][0]), float(existing_pts[best_idx][1])
            new_n = meta[best_idx]['count'] + 1
            avg_lat = (ex_lat * meta[best_idx]['count'] + lat) / new_n
            avg_lng = (ex_lng * meta[best_idx]['count'] + lng) / new_n
            existing_pts[best_idx][0] = avg_lat
            existing_pts[best_idx][1] = avg_lng
            # elevation
            if elev is not None:
                samples = meta[best_idx].get('samples', [])
                samples.append(elev)
                meta[best_idx]['samples'] = samples
                meta[best_idx]['mean_elev'] = float(np.mean(samples))
            meta[best_idx]['count'] = new_n
        else:
            # append as new point and meta
            existing_pts.append([lat, lng])
            meta.append({'samples': [elev] if elev is not None else [], 'count': 1 if elev is not None else 1, 'mean_elev': float(elev) if elev is not None else None})

    # write again
    out = {
        'latlng': {'data': existing_pts, 'series_type': 'distance', 'original_size': len(existing_pts), 'resolution': 'centerline'},
        'latlng_meta': meta
    }
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"[DEBUG] merge_centerline_into_trail: Updated {path} (#points={len(existing_pts)})")
    return out

def assign_points_to_trails(gps_coords, osm_trails_gdf, threshold=25):
    segments = []
    unmatched = []
    current_trail_id = None
    current_points = []

    # try to import shapely Point locally; if not available, abort matching
    try:
        from shapely.geometry import Point
    except Exception:
        print("[DEBUG] shapely not available; assign_points_to_trails will skip matching and return all points as unmatched.")
        return [], [[p[0], p[1], p[2]] if len(p) > 2 else [p[0], p[1]] for p in gps_coords]

    for idx, item in enumerate(tqdm(gps_coords, desc="Assigning points to trails", leave=True, position=0)):
        if idx % 50 == 0:
            print(f"[DEBUG] assign_points_to_trails: processing gps point {idx+1}/{len(gps_coords)}")
        lat = item[0]
        lng = item[1]
        elev = item[2] if len(item) > 2 else None
        pt = Point(lng, lat)

        # find nearest trail linear component
        min_dist = float('inf')
        best_trail_id = None
        best_trail_line = None

        for _, row in osm_trails_gdf.iterrows():
            geom = row.geometry
            candidate_id = str(row.get('id', ''))

            try:
                gtype = geom.geom_type
            except Exception:
                continue

            candidate_line = None
            if gtype in ('LineString', 'LinearRing'):
                candidate_line = geom
            elif gtype == 'MultiLineString':
                try:
                    candidate_line = min(geom.geoms, key=lambda g: pt.distance(g))
                except Exception:
                    candidate_line = None
            elif gtype == 'Polygon':
                try:
                    candidate_line = geom.exterior
                except Exception:
                    candidate_line = None
            elif gtype == 'MultiPolygon':
                try:
                    poly = min(geom.geoms, key=lambda p: pt.distance(p.exterior))
                    candidate_line = poly.exterior
                except Exception:
                    candidate_line = None
            elif gtype == 'GeometryCollection':
                try:
                    linear_parts = [g for g in geom.geoms if g.geom_type in ('LineString', 'LinearRing')]
                    if linear_parts:
                        candidate_line = min(linear_parts, key=lambda g: pt.distance(g))
                except Exception:
                    candidate_line = None

            if candidate_line is None:
                continue

            try:
                dist = pt.distance(candidate_line) * 111139
            except Exception:
                continue

            if dist < min_dist:
                min_dist = dist
                best_trail_id = candidate_id
                best_trail_line = candidate_line

        # threshold check
        if best_trail_line is not None and min_dist <= threshold:
            # begin new segment if trail changed
            if current_trail_id != best_trail_id:
                if current_trail_id is not None and current_points:
                    segments.append((current_trail_id, current_points))
                current_trail_id = best_trail_id
                current_points = []
                print(f"--- New segment started for '{best_trail_id}' ---")

            try:
                proj = best_trail_line.project(pt)
                snap_point = best_trail_line.interpolate(proj)
                snap_lat, snap_lng = snap_point.y, snap_point.x
            except Exception as e:
                print(f"[DEBUG] assign_points_to_trails: failed to snap point {idx+1} to trail {best_trail_id}: {e}")
                continue

            if elev is not None:
                current_points.append([snap_lat, snap_lng, elev])
            else:
                current_points.append([snap_lat, snap_lng])
        else:
            # unmatched point
            unmatched.append([lat, lng, elev] if elev is not None else [lat, lng])
            if current_trail_id is not None and current_points:
                segments.append((current_trail_id, current_points))
                current_trail_id = None
                current_points = []

    if current_trail_id is not None and current_points:
        segments.append((current_trail_id, current_points))

    print(f"[DEBUG] assign_points_to_trails: Total segments found: {len(segments)}; unmatched points: {len(unmatched)}")
    return segments, unmatched

def get_all_raw_points(all_raw_files):
    all_points = []
    for rawfile in all_raw_files:
        with open(rawfile) as f:
            data = json.load(f)
        # Expect "latlng" key with "data" list (hope it works)
        if 'latlng' in data and 'data' in data['latlng']:
            for pt in data['latlng']['data']:
                lat = pt[0]
                lng = pt[1]
                if len(pt) > 2:
                    all_points.append([lat, lng, pt[2]])
                else:
                    all_points.append([lat, lng])
        else:
            print(f"[DEBUG] get_all_raw_points: 'latlng' or 'data' key missing in {rawfile}, skipping.")
            continue
    return all_points


### --- Skeletonization ---
def _get_projection_funcs(sample_points):
    """Return (to_meters, to_latlng, length_units) functions.
    If pyproj is available, return accurate projection; otherwise use simple equirectangular approx.
    length_units is the unit used for LineString lengths/project distances (meters or degrees).
    """
    try:
        from pyproj import Transformer
        # pick central lat/lon
        lats = [p[0] for p in sample_points if p and len(p) >= 2]
        lngs = [p[1] for p in sample_points if p and len(p) >= 2]
        if not lats or not lngs:
            raise Exception('no sample points')
        mean_lat = float(sum(lats) / len(lats))
        mean_lng = float(sum(lngs) / len(lngs))
        # use Azimuthal Equidistant centered at mean (google came in clutch with this one)
        proj_crs = f"+proj=aeqd +lat_0={mean_lat} +lon_0={mean_lng} +units=m +no_defs"
        transformer_to_m = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
        transformer_from_m = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)

        def to_meters_proj(lat, lng):
            x, y = transformer_to_m.transform(lng, lat)
            return x, y

        def to_latlng_proj(x, y):
            lng, lat = transformer_from_m.transform(x, y)
            return lat, lng

        return to_meters_proj, to_latlng_proj, 'm'
    except Exception:
        # fallback: equirctangulr approx using mean latitude
        lats = [p[0] for p in sample_points if p and len(p) >= 2]
        if not lats:
            mean_lat = 0.0
        else:
            mean_lat = float(sum(lats) / len(lats))

        def to_meters(lat, lng):
            x = lng * 111139 * math.cos(math.radians(mean_lat))
            y = lat * 111139
            return x, y

        def to_latlng(x, y):
            lat = y / 111139
            lng = x / (111139 * math.cos(math.radians(mean_lat))) if math.cos(math.radians(mean_lat)) != 0 else 0.0
            return lat, lng

        return to_meters, to_latlng, 'm'


def compute_seed_centerline(runs):
    """Pick a seed centerline from runs. Simple heuristic: choose the run with median length.
    runs: list of lists of [lat,lng(,elev)]
    Returns: list of (lat,lng)
    """
    if not runs:
        return []
    # find lengths
    lens = []
    for r in runs:
        total = 0.0
        for i in range(1, len(r)):
            a = r[i - 1]
            b = r[i]
            total += _haversine_meters(a[0], a[1], b[0], b[1])
        lens.append(total)
    # pick median index
    idx = int(len(runs) / 2)
    try:
        median_idx = sorted(range(len(runs)), key=lambda i: lens[i])[idx]
    except Exception:
        median_idx = 0
    seed = [[p[0], p[1]] for p in runs[median_idx] if len(p) >= 2]
    return seed


def resample_line_by_spacing(points, spacing_m=5.0):
    """Given a list of [lat,lng] points representing a polyline, return resampled points at approx spacing_m meters.
    Uses local projection if available.
    """
    if not points or len(points) < 2:
        return points
    to_meters, to_latlng, units = _get_projection_funcs(points)
    # build metric LineString
    coords_m = [to_meters(lat, lng) for lat, lng in points]
    try:
        from shapely.geometry import LineString, Point
        ls_m = LineString(coords_m)
        length = ls_m.length
        if length <= 0:
            return points
        n = max(1, int(math.floor(length / float(spacing_m))))
        sampled = []
        for i in range(n + 1):
            d = min(length, float(i) * spacing_m)
            pt_m = ls_m.interpolate(d)
            x, y = float(pt_m.x), float(pt_m.y)
            lat, lng = to_latlng(x, y)
            sampled.append([lat, lng])
        return sampled
    except Exception:
        # fallback: linear interpolation by cumulative haversine distance
        cum = [0.0]
        for i in range(1, len(points)):
            a = points[i - 1]
            b = points[i]
            cum.append(cum[-1] + _haversine_meters(a[0], a[1], b[0], b[1]))
        total = cum[-1]
        if total <= 0:
            return points
        n = max(1, int(math.floor(total / spacing_m)))
        samples = []
        for t in [i * spacing_m for i in range(n + 1)]:
            # find segment
            for i in range(1, len(cum)):
                if cum[i] >= t:
                    a = points[i - 1]
                    b = points[i]
                    seg_len = cum[i] - cum[i - 1]
                    if seg_len == 0:
                        frac = 0.0
                    else:
                        frac = (t - cum[i - 1]) / seg_len
                    lat = a[0] + (b[0] - a[0]) * frac
                    lng = a[1] + (b[1] - a[1]) * frac
                    samples.append([lat, lng])
                    break
        return samples


def rdp_simplify(points, tol_m=2.0):
    """Simplify points using RDP-like approach. Prefer shapely.simplify when available (in meters projection).
    points: list of [lat,lng]
    tol_m: tolerance in meters
    """
    if not points or len(points) < 3:
        return points
    # try shapely simplify via projection
    to_meters, to_latlng, units = _get_projection_funcs(points)
    try:
        from shapely.geometry import LineString
        coords_m = [to_meters(lat, lng) for lat, lng in points]
        ls = LineString(coords_m)
        simp = ls.simplify(tol_m, preserve_topology=False)
        out = [to_latlng(float(x), float(y)) for x, y in simp.coords]
        return [[lat, lng] for lat, lng in out]
    except Exception:
        # fallback: implement simple RDP in degrees with tol converted approximately
        # convert tol_m to deg approx
        mean_lat = float(sum([p[0] for p in points]) / len(points))
        tol_deg = tol_m / 111139.0
        # simple recursive RDP
        def _rdp(pts, eps):
            if len(pts) < 3:
                return pts
            a = pts[0]
            b = pts[-1]
            # line AB
            max_dist = 0.0
            idx = 0
            for i in range(1, len(pts) - 1):
                p = pts[i]
                # perpendicular distance to line a-b in degrees using cross product
                num = abs((b[1] - a[1]) * p[0] - (b[0] - a[0]) * p[1] + b[0]*a[1] - b[1]*a[0])
                den = math.hypot(b[1] - a[1], b[0] - a[0])
                d = num / den if den != 0 else 0
                if d > max_dist:
                    idx = i
                    max_dist = d
            if max_dist > eps:
                left = _rdp(pts[:idx+1], eps)
                right = _rdp(pts[idx:], eps)
                return left[:-1] + right
            else:
                return [a, b]
        res = _rdp([[p[0], p[1]] for p in points], tol_deg)
        return res


def detect_branches(seed_s, projected_runs, branch_thresh_m=10.0, min_len_samples=3, window_m=5.0):
    """Detect branches along seed sample distances.
    Returns: list of branch intervals (start_idx,end_idx) and an assignment of runs->branch for each interval.
    Strategy: for each seed sample, cluster projected run points (within window_m) by spatial DBSCAN with eps=branch_thresh_m; if >1 clusters persist for consecutive samples, mark as branch region.
    """
    if not projected_runs or not seed_s:
        return []
    # build buckets per sample index
    from sklearn.cluster import DBSCAN as SKDB
    sample_clusters = []  # list of list of cluster labels per sample (per point -> label)
    points_per_sample = []
    for s in seed_s:
        pts = []
        src = []
        for ridx, pr in enumerate(projected_runs):
            for p in pr:
                if abs(p[0] - s) <= window_m:
                    pts.append((p[1], p[2], ridx))
        if not pts:
            sample_clusters.append([])
            points_per_sample.append([])
            continue
        coords = [(p[0], p[1]) for p in pts]
        if len(coords) == 1:
            labels = [0]
        else:
            try:
                clustering = SKDB(eps=branch_thresh_m, min_samples=1).fit(np.array([[c[0], c[1]] for c in coords]))
                labels = clustering.labels_.tolist()
            except Exception:
                labels = [0] * len(coords)
        sample_clusters.append(labels)
        points_per_sample.append(pts)

    # find consecutive ranges where number of unique clusters >1
    branches = []
    i = 0
    while i < len(seed_s):
        # determine cluster count
        lbls = sample_clusters[i]
        uniq = set(lbls) if lbls else set()
        if len(uniq) > 1:
            j = i
            while j < len(seed_s) and len(set(sample_clusters[j])) > 1:
                j += 1
            if (j - i) >= min_len_samples:
                branches.append((i, j))
            i = j
        else:
            i += 1

    # For each branch interval, assign runs to branch clusters by majority of their points' cluster labels inside interval
    branch_assignments = []  # list per branch: dict run_idx -> cluster_label
    for (s_i, s_j) in branches:
        # collect mapping run_idx -> list of labels
        run_labels = {}
        for si in range(s_i, s_j):
            pts = points_per_sample[si]
            labels = sample_clusters[si]
            for k, t in enumerate(pts):
                lat, lng, ridx = t
                lab = labels[k] if k < len(labels) else 0
                run_labels.setdefault(ridx, []).append(lab)
        # majority vote
        assign = {}
        for ridx, labs in run_labels.items():
            if not labs:
                continue
            # choose most common label
            lbl = max(set(labs), key=labs.count)
            assign[ridx] = lbl
        branch_assignments.append(assign)

    return list(zip(branches, branch_assignments))



def project_run_to_seed(seed_pts, run_pts):
    """Project run points onto seed polyline and return list of (s_dist, lat, lng, elev)
    s_dist uses meters when shapely/pyproj available.
    """
    if not seed_pts or not run_pts:
        return []
    to_meters, to_latlng, units = _get_projection_funcs(seed_pts)
    try:
        from shapely.geometry import LineString, Point
        seed_m = LineString([to_meters(p[0], p[1]) for p in seed_pts])
        projected = []
        for p in run_pts:
            lat, lng = p[0], p[1]
            elev = p[2] if len(p) > 2 else None
            x, y = to_meters(lat, lng)
            pt = Point(x, y)
            s = seed_m.project(pt)
            # convert projected point back to lat/lng
            proj_pt = seed_m.interpolate(s)
            plat, plng = to_latlng(float(proj_pt.x), float(proj_pt.y))
            projected.append((s, plat, plng, elev))
        return projected
    except Exception:
        # fallback: nearest-by-haversine to seed vertices, using cumulative distance as s
        # build cumulative along seed
        cum = [0.0]
        for i in range(1, len(seed_pts)):
            a = seed_pts[i - 1]
            b = seed_pts[i]
            cum.append(cum[-1] + _haversine_meters(a[0], a[1], b[0], b[1]))
        projected = []
        for p in run_pts:
            best = None
            best_d = float('inf')
            for i, sp in enumerate(seed_pts):
                d = _haversine_meters(p[0], p[1], sp[0], sp[1])
                if d < best_d:
                    best_d = d
                    best = (cum[i], sp[0], sp[1])
            elev = p[2] if len(p) > 2 else None
            if best is None:
                continue
            projected.append((best[0], best[1], best[2], elev))
        return projected


def average_along_seed(projected_runs, seed_sample_s, window_m=5.0):
    """For each sample distance in seed_sample_s, average projected run points within window_m.
    projected_runs: list of lists of (s, lat, lng, elev)
    Returns list of [lat,lng,elev?]
    """
    if not projected_runs or not seed_sample_s:
        return []
    # flatten all projected points
    all_pts = []
    for pr in projected_runs:
        for p in pr:
            all_pts.append(p)

    out = []
    for s_target in seed_sample_s:
        bucket = [p for p in all_pts if abs(p[0] - s_target) <= window_m]
        if not bucket:
            out.append([None, None])
            continue
        lats = [p[1] for p in bucket]
        lngs = [p[2] for p in bucket]
        elevs = [p[3] for p in bucket if p[3] is not None]
        avg_lat = float(np.mean(lats))
        avg_lng = float(np.mean(lngs))
        if elevs:
            avg_elev = float(np.mean(elevs))
            out.append([avg_lat, avg_lng, avg_elev])
        else:
            out.append([avg_lat, avg_lng])
    return out


def _skeleton_no_branch(runs, spacing_m=5.0, window_m=5.0):
    """Compute a single averaged skeleton for given runs (no branch splitting).
    Returns a list of [lat,lng(,elev)]."""
    if not runs:
        return []
    seed = compute_seed_centerline(runs)
    if not seed:
        return []
    seed_samples = resample_line_by_spacing(seed, spacing_m=spacing_m)
    projected_runs = [project_run_to_seed(seed_samples, r) for r in runs]
    try:
        max_s = max([max([p[0] for p in pr]) if pr else 0 for pr in projected_runs])
        n = max(1, int(math.floor(max_s / spacing_m)))
        seed_s = [i * spacing_m for i in range(n + 1)]
    except Exception:
        seed_s = [i * spacing_m for i in range(len(seed_samples))]
    averaged = average_along_seed(projected_runs, seed_s, window_m=window_m)
    out = [p for p in averaged if p and p[0] is not None]
    return out


def skeletonize_runs(runs, spacing_m=5.0, simplify_tol_m=2.0, window_m=5.0):
    """Branch-aware skeletonization pipeline.
    Returns a list of skeletons. Each skeleton is a list of [lat,lng(,elev)].
    When no branching is detected, returns a single-item list containing the merged skeleton.
    """
    # normalize input
    if not runs:
        return []

    # first compute a full skeleton and projected runs to detect branches
    seed = compute_seed_centerline(runs)
    if not seed:
        return []
    seed_samples = resample_line_by_spacing(seed, spacing_m=spacing_m)
    projected_runs = [project_run_to_seed(seed_samples, r) for r in runs]

    # build seed cumulative distances sequence (approx)
    try:
        max_s = max([max([p[0] for p in pr]) if pr else 0 for pr in projected_runs])
        n = max(1, int(math.floor(max_s / spacing_m)))
        seed_s = [i * spacing_m for i in range(n + 1)]
    except Exception:
        seed_s = [i * spacing_m for i in range(len(seed_samples))]

    branches = detect_branches(seed_s, projected_runs, branch_thresh_m=spacing_m*2.0, window_m=window_m)
    if not branches:
        # no branching: return single skeleton
        merged = _skeleton_no_branch(runs, spacing_m=spacing_m, window_m=window_m)
        return [merged] if merged else []

    # If branches found: build mainline skeleton excluding branch intervals, and per-branch skeletons
    # branches: list of ((start_idx,end_idx), assignment)
    branch_intervals = [b[0] for b in branches]

    # build mask of seed_s indices that are inside any branch interval
    branch_mask = [False] * len(seed_s)
    for (si, sj) in branch_intervals:
        for k in range(si, min(sj, len(seed_s))):
            branch_mask[k] = True

    # compute mainline using projected_runs but only for seed_s indices not in branch_mask
    all_pts = []
    for pr in projected_runs:
        for p in pr:
            all_pts.append(p)
    main_seed_s = [s for i, s in enumerate(seed_s) if not branch_mask[i]]
    mainline = average_along_seed(projected_runs, main_seed_s, window_m=window_m)
    mainline = [p for p in mainline if p and p[0] is not None]

    result_skeletons = []
    if mainline:
        result_skeletons.append(mainline)

    # For each branch interval produce skeletons per cluster assignment
    for (interval, assignment) in branches:
        s_i, s_j = interval
        # assignment: dict run_idx -> cluster_label
        # invert mapping label -> [run_idx]
        label_map = {}
        for ridx, lbl in assignment.items():
            label_map.setdefault(lbl, []).append(ridx)

        for lbl, run_idxs in label_map.items():
            # build fragment runs: for each run index, extract the original run points whose projected s in [s_i,s_j]
            fragment_runs = []
            for ridx in run_idxs:
                if ridx < 0 or ridx >= len(runs):
                    continue
                orig_run = runs[ridx]
                proj_run = projected_runs[ridx]
                frag = []
                # proj_run entries correspond to orig_run ordering
                # iterate pairs
                # If lengths mismatch, fall back to spatial filter
                if len(proj_run) == len(orig_run):
                    for k, pr in enumerate(proj_run):
                        s_val = pr[0]
                        if s_val >= s_i and s_val <= s_j:
                            frag.append(orig_run[k])
                else:
                    # fallback: include orig points whose nearest seed projection falls within interval
                    for p in orig_run:
                        # find nearest projected entry for this point
                        best_s = None
                        best_d = float('inf')
                        for pr in proj_run:
                            # approximate distance in s-space
                            d = abs(pr[1] - p[0]) + abs(pr[2] - p[1])
                            if d < best_d:
                                best_d = d
                                best_s = pr[0]
                        if best_s is not None and best_s >= s_i and best_s <= s_j:
                            frag.append(p)
                if frag:
                    fragment_runs.append(frag)

            if not fragment_runs:
                continue
            # compute a skeleton for this branch fragment (no further branch splitting)
            frag_sk = _skeleton_no_branch(fragment_runs, spacing_m=spacing_m, window_m=window_m)
            if frag_sk:
                result_skeletons.append(frag_sk)

    return result_skeletons



def process_stream_file(file: str, trails_folder: str = 'trails_db', use_osm: bool = False):
    raw_folder = os.path.join(trails_folder, "raw")
    way_folder = os.path.join(trails_folder, "way")
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(way_folder, exist_ok=True)
    # Always save RAW track first (OSM-independent)
    print(f"[DEBUG] process_stream_file: Loading stream file {file}")
    with open(file, 'r') as f:
        example_streams = json.load(f)

    print(f"[DEBUG] Keys in stream file: {list(example_streams.keys())}")
    latlngs = example_streams.get('latlng', {}).get('data', [])
    altitudes = example_streams.get('altitude', {}).get('data', [])
    print(f"[DEBUG] latlngs count: {len(latlngs)}, altitudes count: {len(altitudes)}")

    gps_coords = []
    if altitudes and len(altitudes) >= len(latlngs):
        gps_coords = [latlngs[i] + [altitudes[i]] for i in range(min(len(latlngs), len(altitudes)))]
    else:
        gps_coords = [p for p in latlngs]
    print(f"[DEBUG] gps_coords count: {len(gps_coords)}")

    raw_trail_id = os.path.splitext(os.path.basename(file))[0] + "_raw"
    latlng_list_raw = [[x[0], x[1]] for x in gps_coords]
    elevation_samples_raw = [x[2] for x in gps_coords if len(x) > 2]
    streams_raw = {
        "latlng": {
            "data": latlng_list_raw,
            "series_type": "distance",
            "original_size": len(latlng_list_raw),
            "resolution": "high"
        }
    }
    print(f"[DEBUG] Saving RAW trail: {raw_trail_id}, points: {len(latlng_list_raw)}")
    save_trail_streams(raw_trail_id, streams_raw, folder=os.path.join(trails_folder, 'raw'), elevation_samples=elevation_samples_raw)

    # Per-run centerline: resample the run into a regular spaced line (preferred over clustering into one point)
    spacing_default = 5.0
    centerline_points = resample_line_by_spacing([[p[0], p[1]] if len(p) >= 2 else p for p in gps_coords], spacing_m=spacing_default)
    # centerline_points now is list of [lat,lng]
    centerline_trail_id = os.path.splitext(os.path.basename(file))[0] + "_centerline"
    latlng_centerline = [[x[0], x[1]] for x in centerline_points]
    elev_centerline = [p[2] for p in gps_coords]  # elevation not preserved per resampled point without projection
    streams_centerline = {
        "latlng": {
            "data": latlng_centerline,
            "series_type": "distance",
            "original_size": len(latlng_centerline),
            "resolution": "centerline"
        }
    }
    print(f"[DEBUG] Saving per-run resampled centerline: {centerline_trail_id}, points: {len(latlng_centerline)}")
    save_trail_streams(centerline_trail_id, streams_centerline, folder=os.path.join(trails_folder, 'way'), elevation_samples=elev_centerline)

    bbox_geojson = os.path.join(trails_folder, 'bbox_raw_saves', 'export.geojson')
    if not use_osm:
        # No-OSM mode (default): merge full centerline into a custom aggregate for this base_id
        custom_id = os.path.splitext(os.path.basename(file))[0] + "_custom_aggregate"
        print(f"[DEBUG] No-OSM mode: merging full centerline into {custom_id}")
        merge_centerline_into_trail(custom_id, centerline_points, folder=os.path.join(trails_folder, 'way'), snap_threshold_m=15)

        # Attempt to find nearby raw runs and produce a merged skeleton if multiple are present
        try:
            # find raw files whose start or centroid is near this run's start
            raw_folder = os.path.join(trails_folder, 'raw')
            if os.path.isdir(raw_folder):
                base_start = gps_coords[0] if gps_coords else None
                merge_candidates = []
                merge_radius_m = 50.0
                for fname in os.listdir(raw_folder):
                    if not fname.endswith('.json'):
                        continue
                    fpath = os.path.join(raw_folder, fname)
                    try:
                        with open(fpath) as rf:
                            d = json.load(rf)
                        pts = d.get('latlng', {}).get('data', [])
                        if not pts:
                            continue
                        start = pts[0]
                        if base_start and _haversine_meters(base_start[0], base_start[1], start[0], start[1]) <= merge_radius_m:
                            # load full run (include elevations if present)
                            alt = d.get('altitude', {}).get('data', []) if 'altitude' in d else []
                            run = []
                            if alt and len(alt) >= len(pts):
                                for i in range(min(len(pts), len(alt))):
                                    run.append([pts[i][0], pts[i][1], alt[i]])
                            else:
                                for p in pts:
                                    run.append([p[0], p[1]])
                            merge_candidates.append(run)
                    except Exception:
                        continue
                if len(merge_candidates) > 1:
                    print(f"[DEBUG] Found {len(merge_candidates)} nearby raw runs — building merged skeleton")
                    merged = skeletonize_runs(merge_candidates, spacing_m=spacing_default)
                    if merged:
                        merged_id = os.path.splitext(os.path.basename(file))[0] + "_merged_skeleton"
                        latlngs_merged = [[p[0], p[1]] for p in merged]
                        elevs_merged = [p[2] for p in merged if len(p) > 2]
                        streams_merged = {"latlng": {"data": latlngs_merged, "series_type": "distance", "original_size": len(latlngs_merged), "resolution": "skeleton"}}
                        print(f"[DEBUG] Saving merged skeleton: {merged_id}, points: {len(latlngs_merged)}")
                        save_trail_streams(merged_id, streams_merged, folder=os.path.join(trails_folder, 'way'), elevation_samples=elevs_merged)
        except Exception as e:
            print(f"[DEBUG] multi-run merge failed: {e}")
        return

    # use_osm == True
    if not os.path.isfile(bbox_geojson):
        print(f"[DEBUG] OSM bbox file not found at {bbox_geojson}. Falling back to No-OSM merge.")
        custom_id = os.path.splitext(os.path.basename(file))[0] + "_custom_aggregate"
        merge_centerline_into_trail(custom_id, centerline_points, folder=os.path.join(trails_folder, 'way'), snap_threshold_m=15)
        return

    # try to import geopandas and shapely when OSM processing is requested
    try:
        import geopandas as gpd
    except Exception as e:
        print(f"[DEBUG] Failed to import geopandas for OSM processing: {e}. Falling back to No-OSM merge.")
        custom_id = os.path.splitext(os.path.basename(file))[0] + "_custom_aggregate"
        merge_centerline_into_trail(custom_id, centerline_points, folder=os.path.join(trails_folder, 'way'), snap_threshold_m=15)
        return

    try:
        osm_trails = gpd.read_file(bbox_geojson)
        print("[DEBUG] OSM columns:", osm_trails.columns)
        print(f"[DEBUG] OSM trails count: {len(osm_trails)}")
    except Exception as e:
        print(f"[DEBUG] Failed to read OSM bbox or merge centerline: {e}")
        custom_id = os.path.splitext(os.path.basename(file))[0] + "_custom_aggregate"
        merge_centerline_into_trail(custom_id, centerline_points, folder=os.path.join(trails_folder, 'way'), snap_threshold_m=15)
        return

    # assign centerline points to OSM segments
    try:
        segments, unmatched = assign_points_to_trails(centerline_points, osm_trails, threshold=25)
    except Exception as e:
        print(f"[DEBUG] assign_points_to_trails failed: {e}. Falling back to No-OSM merge.")
        custom_id = os.path.splitext(os.path.basename(file))[0] + "_custom_aggregate"
        merge_centerline_into_trail(custom_id, centerline_points, folder=os.path.join(trails_folder, 'way'), snap_threshold_m=15)
        return

    print(f"[DEBUG] Segments found: {len(segments)}; Unmatched: {len(unmatched)}")

    # Save unmatched points as a separate artifact (so nothing is lost)
    if unmatched:
        unmatched_id = os.path.splitext(os.path.basename(file))[0] + "_unmatched"
        latlng_list_unmatched = [[p[0], p[1]] for p in unmatched]
        elevation_unmatched = [p[2] for p in unmatched if len(p) > 2]
        streams_unmatched = {
            "latlng": {"data": latlng_list_unmatched, "series_type": "distance", "original_size": len(latlng_list_unmatched), "resolution": "high"}
        }
        print(f"[DEBUG] Saving unmatched points: {unmatched_id}, points: {len(latlng_list_unmatched)}")
        save_trail_streams(unmatched_id, streams_unmatched, folder=trails_folder, elevation_samples=elevation_unmatched)

    if not segments or all(len(seg[1]) == 0 for seg in segments):
        print(f"[DEBUG] No matched OSM segments—(already saved RAW).")
        return

    for idx, (trail_id, points_segment) in enumerate(tqdm(segments, desc="Saving matched segments", leave=True, position=0)):
        print(f"[DEBUG] Processing segment {idx+1}/{len(segments)} for trail_id: {trail_id}, points: {len(points_segment)}")
        row = osm_trails[osm_trails['id'] == trail_id] if 'id' in osm_trails.columns else osm_trails[osm_trails['id'] == int(trail_id)]
        if len(row) == 0:
            print(f"[DEBUG] -- Warning: OSM trail {trail_id} not found, skipping.")
            continue

        latlng_list = [[x[0], x[1]] for x in points_segment]
        elevation_samples = [x[2] for x in points_segment if len(x) > 2]
        streams_segment = {
            "latlng": {"data": latlng_list, "series_type": "distance", "original_size": len(latlng_list), "resolution": "high"}
        }
        print(f"[DEBUG] Saving segment {idx+1}/{len(segments)} for trail '{trail_id}' with {len(latlng_list)} points...")
        save_trail_streams(trail_id, streams_segment, folder=trails_folder, elevation_samples=elevation_samples)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-stream', help='Path to a stream JSON file to process and save into trails_db')
    parser.add_argument('--trails-folder', default='trails_db', help='Folder where trails_db is located')
    parser.add_argument('--osm', action='store_true', help='Enable OSM matching (requires geopandas/shapely)')
    parser.add_argument('--skeletonize', action='store_true', help='Run skeletonization/resampling on the provided stream')
    parser.add_argument('--spacing', type=float, default=5.0, help='Spacing in meters for skeleton resampling')
    parser.add_argument('--window', type=float, default=5.0, help='Projection window in meters when averaging')
    args = parser.parse_args()
    if args.process_stream:
        process_stream_file(args.process_stream, trails_folder=args.trails_folder, use_osm=bool(args.osm))
        if args.skeletonize:
            # Load the stream and run skeletonize on this single run (or extend to multiple raw files later)
            with open(args.process_stream, 'r') as f:
                data = json.load(f)
            latlngs = data.get('latlng', {}).get('data', [])
            altitudes = data.get('altitude', {}).get('data', [])
            if altitudes and len(altitudes) >= len(latlngs):
                gps = [latlngs[i] + [altitudes[i]] for i in range(min(len(latlngs), len(altitudes)))]
            else:
                gps = [p for p in latlngs]
            runs = [gps]
            print(f"[DEBUG] Running skeletonize on {len(runs)} runs (spacing={args.spacing}m, window={args.window}m)")
            sk = skeletonize_runs(runs, spacing_m=args.spacing, window_m=args.window)
            base_id = os.path.splitext(os.path.basename(args.process_stream))[0]
            sk_id = base_id + "_skeleton"
            if sk:
                latlngs_sk = [[p[0], p[1]] for p in sk]
                elevs_sk = [p[2] for p in sk if len(p) > 2]
                streams_sk = {"latlng": {"data": latlngs_sk, "series_type": "distance", "original_size": len(latlngs_sk), "resolution": "skeleton"}}
                print(f"[DEBUG] Saving skeleton centerline: {sk_id}, points: {len(latlngs_sk)}")
                save_trail_streams(sk_id, streams_sk, folder=os.path.join(args.trails_folder, 'way'), elevation_samples=elevs_sk)
            else:
                print("[DEBUG] Skeletonization returned no points.")

