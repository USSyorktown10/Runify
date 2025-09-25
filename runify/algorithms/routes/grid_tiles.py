import geopandas as gpd
import glob
import pandas as pd

def merge_and_deduplicate_geojson_files(tile_files, output_file='nc_merged.geojson'):
    dfs = [gpd.read_file(f) for f in tile_files]
    merged = pd.concat(dfs, ignore_index=True)
    idcol = '@id' if '@id' in merged.columns else 'id'
    deduped = merged.drop_duplicates(subset=[idcol])
    deduped_gdf = gpd.GeoDataFrame(deduped, geometry='geometry', crs=dfs[0].crs)
    deduped_gdf.to_file(output_file, driver='GeoJSON')
    print(f"Merged {len(tile_files)} files, unique features: {len(deduped_gdf)}")

# Usage example
tile_files = glob.glob('trails_db/bbox_raw_saves/export(2).geojson')
merge_and_deduplicate_geojson_files(tile_files, 'nc_merged.geojson')
