from ntpath import join
import math

import geopandas as gpd
import pandas as pd
import rasterio as rio
import morecantile
import pyproj

import requests
import lxml
import os, sys
import argparse
from tqdm import tqdm
import yaml



# Get the configuration
# parser = argparse.ArgumentParser(description="This script prepares datasets for the determination of the road cover type.")
# parser.add_argument('config_file', type=str, help='a YAML config file')
# args = parser.parse_args()

with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['prepare_data.py']    #  [os.path.basename(__file__)]


# Task to do
DETERMINE_ROAD_SURFACES = cfg['tasks']['determine_roads_surfaces']
DETERMINE_RESTRICTED_AOI = cfg['tasks']['determine_restricted_AOI']
MAKE_RASTER_MOSAIC = cfg['tasks']['make_raster_mosaic']
GENERATE_TILES_JSON=cfg['tasks']['generate_tiles_json']

if not DETERMINE_ROAD_SURFACES and not DETERMINE_RESTRICTED_AOI and not MAKE_RASTER_MOSAIC and not DETERMINE_TILES:
    print('Nothing to do. Exiting!')
    sys.exit(0)
else:

    INPUT = cfg['input']
    INPUT_DIR =INPUT['input_folder']

    ROADS_IN = INPUT_DIR + INPUT['input_files']['roads']
    ROADS_PARAM = INPUT_DIR + INPUT['input_files']['roads_param']
    FORESTS = INPUT_DIR + INPUT['input_files']['forests']
    TILES_SWISSIMAGES = INPUT['input_files']['tiles_swissimages10']

    OUTPUT = cfg['output']
    OUTPUT_DIR = OUTPUT['output_folder']

    if DETERMINE_ROAD_SURFACES:
        ROADS_OUT = OUTPUT_DIR +  OUTPUT['output_files']['roads']

    if DETERMINE_RESTRICTED_AOI:
        RESTRICTED_AOI = OUTPUT_DIR +  OUTPUT['output_files']['restricted_AOI']
    
    if GENERATE_TILES_JSON:
        TILES_AOI = OUTPUT_DIR + OUTPUT['output_files']['tiles_aoi']
        GPKG_TILES= OUTPUT_DIR + OUTPUT['output_files']['gpkg_tiles']
        ZOOM_LEVEL=cfg['param_tiles']['zoom_level']


# Define functions --------------------------------------------------------------------

def polygons_diff_without_artifacts(polygons, p1_idx, p2_idx):
    # Make the difference of the geometry at row p2_idx with the one at the row p1_idx
    
    # Store intermediary results back to poly
    diff=polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']

    if diff.geom_type == 'Polygon':
        polygons.loc[p2_idx,'geometry'] -= polygons.loc[p1_idx,'geometry']

    elif diff.geom_type == 'MultiPolygon':
        # if a multipolygone is created, only keep the largest part to avoid the following error: https://github.com/geopandas/geopandas/issues/992
        polygons.loc[p2_idx,'geometry'] = max((polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']).geoms, key=lambda a: a.area)

    return polygons

def test_crs(crs1, crs2 = "EPSG:2056"):
    try:
        assert(crs1 == crs2), "CRS mismatch between the roads file and the forests file."
    except Exception as e:
        print(e)
        sys.exit(1)

# Import files ------------------------------------------------------------------------------------------
print('Importing files...')

## Data
roads=gpd.read_file(ROADS_IN)
forests=gpd.read_file(FORESTS)
tiles_swissimages=gpd.read_file(TILES_SWISSIMAGES)

## Other informations
roads_parameters=pd.read_excel(ROADS_PARAM)

print('Importations done!')

# Information treatment -----------------------------------------------------------------------------------------

# Filter the roads to consider
roads_parameters=roads_parameters[roads_parameters['to keep']=='yes']
roads_parameters.drop_duplicates(subset='GDB-Code',inplace=True)       # Keep first by default 

joined_roads=roads.merge(roads_parameters[['GDB-Code','Width']], how='right',left_on='OBJEKTART',right_on='GDB-Code')


if DETERMINE_ROAD_SURFACES:
    print('Determining the surface of the roads from lines...')

    # Buffer the roads
    print('Buffering the roads...')

    buffered_roads=joined_roads.copy()
    buffered_roads['buffered_geom']=buffered_roads.buffer(joined_roads['Width'], cap_style=2)

    buffered_roads.drop(columns=['geometry'],inplace=True)
    buffered_roads.rename(columns={'buffered_geom':'geometry'},inplace=True)

    ## Do not let roundabout parts make artifacts
    for idx in buffered_roads.index:
        geom=buffered_roads.loc[idx,'geometry']
        if geom.geom_type == 'MultiPolygon':
            buffered_roads.loc[idx,'geometry'] = max(buffered_roads.loc[idx,'geometry'].geoms, key=lambda a: a.area)


    # Erase overlapping zones of roads buffer
    print('--- Comparing roads for intersections to remove...')

    ## For roads of different width
    print('----- Removing overlap between roads of different classes...')

    buffered_roads['saved_geom']=buffered_roads.geometry
    joined_roads=gpd.sjoin(buffered_roads,buffered_roads[['OBJECTID','OBJEKTART','saved_geom','geometry']],how='left', lsuffix='1', rsuffix='2')

    ### Drop excessive rows
    intersected=joined_roads[joined_roads['OBJECTID_2'].notna()].copy()
    intersected_not_itself=intersected[intersected['OBJECTID_1']!=intersected['OBJECTID_2']].copy()
    intersected_roads=intersected_not_itself.drop_duplicates(subset=['OBJECTID_1','OBJECTID_2'])

    intersected_roads.reset_index(inplace=True, drop=True)

    ### Sort the roads so that the widest ones come first
    ### TO DO : automatize this part for when different object classes are kept
    intersected_roads.loc[intersected_roads['OBJEKTART_1']==20,'OBJEKTART_1']=8.5

    intersect_other_width=intersected_roads[intersected_roads['OBJEKTART_1']<intersected_roads['OBJEKTART_2']].copy()

    intersect_other_width.sort_values(by=['OBJEKTART_1'],inplace=True)
    intersect_other_width.loc[intersect_other_width['OBJEKTART_1']==8.5,'OBJEKTART_1']=20

    intersect_other_width.reset_index(inplace=True, drop=True)

    ### Suppress the overlapping intersection
    ### from https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
    corr_overlap1 = buffered_roads.copy()

    for idx in tqdm(intersect_other_width.index, total=intersect_other_width.shape[0],
                desc='Suppressing the overlap of roads with different width'):
        
        poly1_id = corr_overlap1.index[corr_overlap1['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_1']].values.astype(int)[0]
        poly2_id = corr_overlap1.index[corr_overlap1['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_2']].values.astype(int)[0]
        
        corr_overlap1=polygons_diff_without_artifacts(corr_overlap1,poly1_id,poly2_id)

    corr_overlap1.drop(columns=['saved_geom'],inplace=True)
        
    ## Remove overlapping area between roads of the same class
    print('----- Removing overlap between roads of the same class...')

    save_geom=corr_overlap1.copy()
    save_geom['saved_geom']=save_geom.geometry
    joined_roads=gpd.sjoin(save_geom,save_geom[['OBJECTID','saved_geom','geometry']],how='left', lsuffix='1', rsuffix='2')

    ### Drop excessive rows
    intersected=joined_roads[joined_roads['OBJECTID_2'].notna()].copy()
    intersected_not_itself=intersected[intersected['OBJECTID_1']!=intersected['OBJECTID_2']].copy()
    intersected_roads=intersected_not_itself.drop_duplicates(subset=['OBJECTID_1','OBJECTID_2'])

    intersected_roads.reset_index(inplace=True, drop=True)

    ### Get rid of duplicates not on the same row
    to_drop=[]
    for idx in tqdm(intersected_roads.index, total=intersected_roads.shape[0],
                desc='Ereasing duplicates from spatial join'):
        ir1_objid=intersected_roads.loc[idx,'OBJECTID_1']
        ir2_objid=intersected_roads.loc[idx,'OBJECTID_2']
        
        for ss_idx in intersected_roads[intersected_roads['OBJECTID_1']==ir2_objid].index:
            
            if ir1_objid==intersected_roads.loc[ss_idx,'OBJECTID_2'] and idx<ss_idx:
                to_drop.append(ss_idx)

    intersected_roads.drop(to_drop,inplace=True)

    corr_overlap2=corr_overlap1.copy()

    ### from https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
    for idx in tqdm(intersected_roads.index, total=intersected_roads.shape[0],
                    desc='Suppressing overlap between equivalent roads'):
        
        poly1_id = corr_overlap2.index[corr_overlap2['OBJECTID'] == intersected_roads.loc[idx,'OBJECTID_1']].values.astype(int)[0]
        poly2_id = corr_overlap2.index[corr_overlap2['OBJECTID'] == intersected_roads.loc[idx,'OBJECTID_2']].values.astype(int)[0]
        
        geom1 = corr_overlap2.loc[poly1_id,'geometry']
        geom2 = corr_overlap2.loc[poly2_id,'geometry']

        # Store intermediary results in variable
        diff = geom2 - geom1
        
        if diff.geom_type == 'Polygon':
            temp = geom2 - geom1
            
        elif diff.geom_type == 'MultiPolygon':
            # if a multipolygone is created, only keep the largest part to avoid the following error: https://github.com/geopandas/geopandas/issues/992
            temp = max((geom2 - geom1).geoms, key=lambda a: a.area)

        corr_overlap2=polygons_diff_without_artifacts(corr_overlap2,poly2_id,poly1_id)

        corr_overlap2.loc[poly2_id,'geometry']=temp

    # Exclude the roads potentially under forest canopy
    print('Excluding roads under forest canopy ...')

    test_crs(corr_overlap2.crs, forests.crs)

    non_forest_roads=corr_overlap2.copy()
    non_forest_roads=non_forest_roads.overlay(forests[['UUID','geometry']],how='difference')

    non_forest_roads.drop(columns=['UUID','GDB-Code'],inplace=True)

    print('Done determining the surface of the roads from lines!')


if DETERMINE_RESTRICTED_AOI:
    print('Determing the restricted AOI around the considered roads...')

    print('Calculating buffer...')
    width=roads_parameters['Width'].max()+1

    buffered_roads_aoi=joined_roads.copy()
    buffered_roads_aoi['buffered_geom']=buffered_roads_aoi.buffer(width)
    
    buffered_roads_aoi.drop(columns=['geometry'],inplace=True)
    buffered_roads_aoi.rename(columns={'buffered_geom':'geometry'},inplace=True)

    AOI_roads=buffered_roads_aoi.unary_union

    print('Excluding parts under forest canopy...')
    geom={'geometry':[x for x in AOI_roads.geoms]}
    AOI_roads_no_forest=gpd.GeoDataFrame(geom, crs=roads.crs)

    test_crs(AOI_roads_no_forest.crs, forests.crs)
        
    AOI_roads_no_forest=AOI_roads_no_forest.overlay(forests[['UUID','geometry']],how='difference')

    print('Done determining the restricted AOI!')


if MAKE_RASTER_MOSAIC:
    print('Making the raster mosaic from drone pictures...')

    print('Done making the raster mosaic!')


if GENERATE_TILES_JSON:
    print('Downloading tiles from map.geo.admin WMS...')

    if not DETERMINE_RESTRICTED_AOI:
        AOI_roads_no_forest = gpd.read_file(OUTPUT_DIR +  OUTPUT['output_files']['restricted_AOI'])
    
    bboxes_extent_4326=AOI_roads_no_forest.to_crs(epsg=4326).unary_union.bounds

    # cf. https://developmentseed.org/morecantile/usage/
    crs_4326 = pyproj.CRS.from_epsg(4327)

    tms = morecantile.tms.get("WebMercatorQuad")

    epsg4326_tiles_gdf = gpd.GeoDataFrame.from_features([tms.feature(x, projected=True) for x in tms.tiles(*bboxes_extent_4326, zooms=[ZOOM_LEVEL])])
    epsg3857_tiles_gdf= epsg4326_tiles_gdf.set_crs(epsg=3857)

    AOI_roads_no_forest.to_crs(epsg=3857,inplace=True)
    AOI_roads_no_forest.rename(columns={'FID': 'id_aoi'},inplace=True)

    test_crs(epsg3857_tiles_gdf.crs, AOI_roads_no_forest.crs)

    tiles_in_restricted_aoi=gpd.sjoin(epsg3857_tiles_gdf,AOI_roads_no_forest, how='inner')

    print('Done downloading tiles!')

# Save results ------------------------------------------------------------------
print('Saving files...')

written_files=[]

if DETERMINE_ROAD_SURFACES:
    non_forest_roads.to_file(ROADS_OUT)
    written_files.append(ROADS_OUT)

if DETERMINE_RESTRICTED_AOI:
    AOI_roads_no_forest.to_file(RESTRICTED_AOI)
    written_files.append(RESTRICTED_AOI)

if GENERATE_TILES_JSON:
    # Save in json format for futur use
    tiles_in_restricted_aoi.to_file(TILES_AOI, driver='GeoJSON')

    # Save in gpkg for information
    layername="epsg3857_z" + str(ZOOM_LEVEL) + "_tiles"
    tiles_in_restricted_aoi.to_file(GPKG_TILES, driver='GPKG',layer=layername)

    print('All done!')
    print(f'Written files: {written_files}')
