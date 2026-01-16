"""Rebuild averaged skeletons from raw runs in trails_db/raw.
Groups nearby raw runs by centroid (DBSCAN with eps=50m) and runs skeletonize_runs on each group.
Saves results to trails_db/way/<group_base>_merged_skeleton.json
"""
import os
import json
import math
import numpy as np
from sklearn.cluster import DBSCAN
import importlib.util
import sys
import argparse
import datetime
THIS_DIR = os.path.dirname(__file__)

# ai logs folder
AI_LOGS = os.path.join(THIS_DIR, 'ai_logs')
os.makedirs(AI_LOGS, exist_ok=True)
AI_LOG = os.path.join(AI_LOGS, 'gemini.log')

RAW_DIR = os.path.join(THIS_DIR, 'trails_db', 'raw')
WAY_DIR = os.path.join(THIS_DIR, 'trails_db', 'way')

# import route_save module from same folder
spec = importlib.util.spec_from_file_location('route_save', os.path.join(THIS_DIR, 'route_save.py'))
if spec is None or spec.loader is None:
    print('Failed to load route_save spec from', os.path.join(THIS_DIR, 'route_save.py'))
    sys.exit(1)
route_save = importlib.util.module_from_spec(spec)  # type: ignore
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
def _log_ai(prompt, response, action=None):
    # Robust AI logger: attempt to extract a readable text from SDK response objects,
    # fall back to repr(response), and write a JSON line for easier parsing.
    try:
        raw_text = None
        try:
            if isinstance(response, str):
                raw_text = response
            else:
                # Try common SDK shapes: candidates -> content -> parts -> text
                cands = getattr(response, 'candidates', None)
                if cands:
                    try:
                        first = cands[0]
                        content = getattr(first, 'content', None)
                        if content:
                            parts = getattr(content, 'parts', None)
                            if parts and len(parts) > 0:
                                raw_text = getattr(parts[0], 'text', None)
                    except Exception:
                        raw_text = None
                # fallback: direct text attribute
                if raw_text is None:
                    raw_text = getattr(response, 'text', None)
        except Exception:
            raw_text = None

        if raw_text is None:
            raw_text = repr(response)

        # try to parse JSON from the raw_text when possible
        response_parsed = None
        try:
            # strip fenced code blocks if present
            t = raw_text.strip()
            if t.startswith('```') and 'json' in t[:10]:
                # remove leading ```json and trailing ```
                try:
                    inner = t.split('```', 2)[2].strip()
                except Exception:
                    inner = t
                t = inner
            # attempt JSON parse
            try:
                response_parsed = json.loads(t)
            except Exception:
                response_parsed = None
        except Exception:
            response_parsed = None

        entry = {
            'ts': datetime.datetime.utcnow().isoformat(),
            'prompt': prompt,
            'response_raw': raw_text,
            'response_parsed': response_parsed,
        }
        if action is not None:
            entry['action'] = action
        # if caller already parsed the response, include it under 'parsed'
        # (caller can choose to json.dumps parsed if needed)
        # Write as JSONL (one JSON object per line)
        try:
            with open(AI_LOG, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write ai log: {e}", file=sys.stderr)
    except Exception as e:
        # ensure we never raise from logging
        print(f"[ERROR] _log_ai general failure: {e}", file=sys.stderr)

# optional Gemini integration to recommend merge radius (overridden when --ai-review not set)
default_eps = 50.0
try:
    from gemini_client import suggest_merge_group, analyze_skeleton
except Exception:
    suggest_merge_group = lambda *a, **k: None
    analyze_skeleton = lambda *a, **k: None

# cluster centroids
XY = np.asarray([[x, y] for x, y in coords_m], dtype=float)
cl = DBSCAN(eps=default_eps, min_samples=1).fit(XY)
labels = cl.labels_
labels = cl.labels_

groups = {}
for i, lbl in enumerate(labels):
    groups.setdefault(int(lbl), []).append(raw_runs[i])

print(f'Found {len(groups)} groups for {len(raw_runs)} raw runs')

os.makedirs(WAY_DIR, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--ai-review', action='store_true', help='Enable Gemini AI review for merge decisions')
parser.add_argument('--only-raws', nargs='*', help='Optional list of raw run JSON files to restrict processing to')
args = parser.parse_args()

# If --only-raws provided, override raw_files collected earlier
if args.only_raws:
    provided = []
    for p in args.only_raws:
        if os.path.isabs(p):
            provided.append(p)
        else:
            provided.append(os.path.join(THIS_DIR, p))
    raw_files = [p for p in provided if os.path.isfile(p)]
    if not raw_files:
        print('No provided raw files found; nothing to do.')
        sys.exit(0)

for lbl, items in groups.items():
    print(f'Group {lbl}: {len(items)} runs')
    runs = [it['run'] for it in items]
    # optionally consult Gemini to decide whether to merge, and to override eps
    merge_allowed = True
    ai_decision = None
    try:
        if args.ai_review:
            summary = [{'base': r['base'], 'len': len(r['run'])} for r in items]
            gem = suggest_merge_group(summary)
            # log raw and parsed AI response; action will be filled after reading gem
            try:
                _log_ai(json.dumps(summary), json.dumps(gem), action='ai-suggest-merge')
            except Exception:
                _log_ai(json.dumps(summary), str(gem), action='ai-suggest-merge')
            if gem is not None:
                if isinstance(gem, dict) and gem.get('merge') is False:
                    merge_allowed = False
                    _log_ai(json.dumps(summary), json.dumps(gem), action='skip-merge')
                if isinstance(gem, dict) and gem.get('merge_radius_m') is not None:
                    try:
                        mr = gem.get('merge_radius_m')
                        default_eps = float(str(mr))
                    except Exception:
                        pass
                    _log_ai(json.dumps(summary), json.dumps(gem), action=f'override-eps:{gem.get("merge_radius_m")}')
            # store AI decision for later annotation
            ai_decision = gem
    except Exception as e:
        print('  AI review failed:', e)

    if not merge_allowed:
        print('  AI indicated to skip merge for this group')
        continue

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
            # preserve elevation per point when present
            latlngs = [([p[0], p[1], p[2]] if len(p) > 2 else [p[0], p[1]]) for p in merged]
            elevs = [p[2] for p in merged if len(p) > 2]
            streams = {"latlng": {"data": latlngs, "series_type": "distance", "original_size": len(latlngs), "resolution": "skeleton"}}
            # optional AI analysis/annotation
            try:
                ai = analyze_skeleton(merged)
                # attach parsed ai info to streams for auditability
                if ai is not None:
                    streams['_ai_parsed'] = ai
                    _log_ai(json.dumps({'merged_points': len(merged), 'group': lbl}), json.dumps(ai), action='ai-analyze-skeleton')
            except Exception as e:
                # log ai analysis failure
                _log_ai(json.dumps({'merged_points': len(merged), 'group': lbl}), f'ANALYZE_FAILED: {e}', action='ai-analyze-failed')

            # attach the merge decision (if any) returned earlier by suggest_merge_group
            try:
                if 'ai_decision' in locals() and ai_decision is not None:
                    streams['_ai_decision'] = ai_decision
            except Exception:
                pass

            # attach raw AI text if gemini_client exposes LAST_RAW
            try:
                import gemini_client
                raw = getattr(gemini_client, 'LAST_RAW', None)
                if raw:
                    streams['_ai_raw'] = raw
            except Exception:
                pass

            route_save.save_trail_streams(out_id, streams, folder=WAY_DIR, elevation_samples=elevs)
            print('  saved', out_id, 'points:', len(latlngs))
    except Exception as e:
        print('  failed to merge group', lbl, e)

print('Done')
