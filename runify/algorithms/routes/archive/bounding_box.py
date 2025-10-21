import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box as shapely_box

gdf = gpd.read_file('osm_trails_large.geojson')

minx, miny, maxx, maxy = gdf.total_bounds

print(f"Bounding box: lat=({miny}, {maxy})  lng=({minx}, {maxx})")

box = {
    'min_lat': miny,
    'max_lat': maxy,
    'min_lng': minx,
    'max_lng': maxx
}
print("Bounding box dictionary:", box)

bbox = shapely_box(minx, miny, maxx, maxy)
base = gdf.plot(color='blue', alpha=0.5)
gpd.GeoSeries([bbox]).plot(ax=base, color='red', alpha=0.2, edgecolor='black')
plt.show()
