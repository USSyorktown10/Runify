"""
Load trails and display them
"""

from flask import Flask, render_template, jsonify
import os, json

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('map.html')

@app.route('/api/trails')
def get_trails():
    all_trails = []
    directory = 'trails_db/way'  

    for fname in os.listdir(directory):
        if fname.endswith('.json') and not fname.__contains__('raw'):
            print(fname)
            with open(os.path.join(directory, fname)) as f:
                j = json.load(f)
                latlngs = j.get('latlng', {}).get('data', [])
                elev = j.get('elevation_stats', {}).get('samples', [])

                # Patch: zip elevation values only (not lat/lng pairs)
                latlng_elev = []
                # If elevation is a list of numbers, zip as [lat, lng, elev]
                if latlngs and elev and all(isinstance(e, (int, float)) for e in elev) and len(latlngs) == len(elev):
                    for pt, e in zip(latlngs, elev):
                        latlng_elev.append(pt + [e])
                else:
                    for i, pt in enumerate(latlngs):
                        if i < len(elev) and isinstance(elev[i], (int, float)):
                            latlng_elev.append(pt + [elev[i]])
                        else:
                            latlng_elev.append(pt)

                all_trails.append({
                    'trail_id': fname[:-5],
                    'latlng': latlng_elev,
                    'segments': j.get('segments', [])
                })
    return jsonify(all_trails)

'''
@app.route('/api/average_trails')
def get_average_trails():
    directory = 'your_centerline_results'
    all_centerlines = []
    for fname in os.listdir(directory):
        if fname.endswith('.geojson'):
            with open(os.path.join(directory, fname)) as f:
                gj = json.load(f)
                all_centerlines.append(gj)
    return jsonify(all_centerlines)
'''

# Optional: Load OSM GeoJSON
@app.route('/api/osm_trails')
def get_osm_trails():
    with open('trails_db/bbox_raw_saves/export.geojson') as f:
        return f.read()

if __name__ == '__main__':
    app.run(debug=True)
