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
        if fname.endswith('.json'):
            with open(os.path.join(directory, fname)) as f:
                j = json.load(f)
                all_trails.append({
                    'trail_id': fname[:-5],                         
                    'latlng': j.get('latlng', {}).get('data', []),  
                    'segments': j.get('segments', [])
                })
    return jsonify(all_trails)


# Optional: Load OSM GeoJSON
@app.route('/api/osm_trails')
def get_osm_trails():
    with open('trails_db/bbox_raw_saves/export.geojson') as f:
        return f.read()

if __name__ == '__main__':
    app.run(debug=True)
