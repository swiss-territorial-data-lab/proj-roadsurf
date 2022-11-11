import math
import re

import geopandas as gpd
import pandas as pd
import morecantile

import os, sys
import argparse
from tqdm import tqdm
import yaml

import fct_misc


# Get the configuration
# parser = argparse.ArgumentParser(description="This script prepares datasets for the determination of the road cover type.")
# parser.add_argument('config_file', type=str, help='a YAML config file')
# args = parser.parse_args()

with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['prepare_data.py']    #  [os.path.basename(__file__)]


# Task to do
DETERMINE_ROAD_SURFACES = cfg['tasks']['determine_roads_surfaces']
DETERMINE_RESTRICTED_AOI = cfg['tasks']['determine_restricted_AOI']
GENERATE_TILES_INFO=cfg['tasks']['generate_tiles_info']
GENERATE_LABELS=cfg['tasks']['generate_labels']

if not (DETERMINE_ROAD_SURFACES or DETERMINE_RESTRICTED_AOI or GENERATE_TILES_INFO or GENERATE_LABELS) :
    print('Nothing to do. Exiting!')
    sys.exit(0)
else:

    INPUT = cfg['input']
    INPUT_DIR =INPUT['input_folder']

    ROADS_IN = INPUT_DIR + INPUT['input_files']['roads']
    ROADS_PARAM = INPUT_DIR + INPUT['input_files']['roads_param']
    FORESTS = INPUT_DIR + INPUT['input_files']['forests']
    AOI = INPUT_DIR + INPUT['input_files']['aoi']

    OUTPUT_DIR = cfg['output_folder']
    
    DEBUG_MODE=cfg['debug_mode']
    KUNSTBAUTE_TO_KEEP=[100, 200]
    BELAGSART_TO_KEEP=[100, 200]

    if GENERATE_TILES_INFO or GENERATE_LABELS:
        ZOOM_LEVEL=cfg['zoom_level']

path_shp_gpkg=fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'shapefiles_gpkg'))
path_json=fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR,'json'))

# Define functions --------------------------------------------------------------------

def polygons_diff_without_artifacts(polygons, p1_idx, p2_idx):
    '''
    Make the difference of the geometry at row p2_idx with the one at the row p1_idx
    
    - dataset of polygons
    - index of the "obstacle" polygon in the dataset
    - index of the final polygon
    '''
    
    # Store intermediary results back to poly
    diff=polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']

    if diff.geom_type == 'Polygon':
        polygons.loc[p2_idx,'geometry'] -= polygons.loc[p1_idx,'geometry']

    elif diff.geom_type == 'MultiPolygon':
        # if a multipolygone is created, only keep the largest part to avoid the following error: https://github.com/geopandas/geopandas/issues/992
        polygons.loc[p2_idx,'geometry'] = max((polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']).geoms, key=lambda a: a.area)

        parts=[poly.area for poly in diff.geoms]
        parts.sort(reverse=True)
        
        for area in parts[1:]:
            if area>3:
                print(f"WARNING: when filtering for multipolygons, an area of {round(area,2)} m2 was lost for the polygon {round(polygons.loc[p2_idx,'OBJECTID'])}.")
                # To correct that, we should introduce a second id that we could prolong with the "new" road made by the multipolygon parts, while
                # maintaining the orginal id to trace the roads back at the end.
                # Or add .1, .2 ect. to the original 1 
                # Or to only have multipolygons to pass the function gpd.overlay
                # Or add id based on geometry

    return polygons


# Information treatment -----------------------------------------------------------------------------------------

if DETERMINE_ROAD_SURFACES or DETERMINE_RESTRICTED_AOI:

    print('Importing files...')

    ## Geodata
    roads=gpd.read_file(ROADS_IN)
    forests=gpd.read_file(FORESTS)
    aoi=gpd.read_file(AOI)

    ## Other informations
    roads_parameters=pd.read_excel(ROADS_PARAM)

    # Filter the roads to consider
    print('Filtering the considered roads...')
    roads_parameters=roads_parameters[roads_parameters['to keep']=='yes']
    roads_parameters.drop_duplicates(subset='GDB-Code',inplace=True)       # Keep first by default 

    joined_roads=roads.merge(roads_parameters[['GDB-Code','Width']], how='right',left_on='OBJEKTART',right_on='GDB-Code')
    joined_uncovered_roads=joined_roads[joined_roads['KUNSTBAUTE'].isin(KUNSTBAUTE_TO_KEEP)]
    joined_uncovered_roads_of_interest=joined_uncovered_roads[joined_uncovered_roads['BELAGSART'].isin(BELAGSART_TO_KEEP)]

    aoi_geom=gpd.GeoDataFrame({'id': [0], 'geometry': [aoi['geometry'].unary_union]}, crs=2056)
    fct_misc.test_crs(joined_uncovered_roads_of_interest.crs, aoi_geom.crs)
    joined_roads_in_aoi=joined_uncovered_roads_of_interest.overlay(aoi_geom, how='intersection')

    if DEBUG_MODE:
        joined_roads_in_aoi=joined_roads_in_aoi[1:100]
    

if DETERMINE_ROAD_SURFACES:
    print('Determining the surface of the roads from lines...')

    # Buffer the roads
    print('-- Buffering the roads...')

    buffered_roads=joined_roads_in_aoi.copy()
    buffered_roads['buffered_geom']=buffered_roads.buffer(joined_roads_in_aoi['Width']/2, cap_style=2)

    buffered_roads.drop(columns=['geometry'],inplace=True)
    buffered_roads.rename(columns={'buffered_geom':'geometry'},inplace=True)

    ## Do not let roundabout parts make artifacts
    for idx in buffered_roads.index:
        geom=buffered_roads.loc[idx,'geometry']
        if geom.geom_type == 'MultiPolygon':
            buffered_roads.loc[idx,'geometry'] = max(buffered_roads.loc[idx,'geometry'].geoms, key=lambda a: a.area)


    # Erase overlapping zones of roads buffer
    print('-- Comparing roads for intersections to remove...')

    ## For roads of different width
    print('----- Removing overlap between roads of different classes...')

    buffered_roads['saved_geom']=buffered_roads.geometry
    joined_roads_in_aoi=gpd.sjoin(buffered_roads,buffered_roads[['OBJECTID','OBJEKTART','saved_geom','geometry']],how='left', lsuffix='1', rsuffix='2')

    ### Drop excessive rows
    intersected=joined_roads_in_aoi[joined_roads_in_aoi['OBJECTID_2'].notna()].copy()
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
                desc='-- Suppressing the overlap of roads with different width'):
        
        poly1_id = corr_overlap1.index[corr_overlap1['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_1']].values.astype(int)[0]
        poly2_id = corr_overlap1.index[corr_overlap1['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_2']].values.astype(int)[0]
        
        corr_overlap1=polygons_diff_without_artifacts(corr_overlap1,poly1_id,poly2_id)

    corr_overlap1.drop(columns=['saved_geom'],inplace=True)
        
    ## Remove overlapping area between roads of the same class
    print('----- Removing overlap between roads of the same class...')

    save_geom=corr_overlap1.copy()
    save_geom['saved_geom']=save_geom.geometry
    joined_roads_in_aoi=gpd.sjoin(save_geom,save_geom[['OBJECTID','saved_geom','geometry']],how='left', lsuffix='1', rsuffix='2')

    ### Drop excessive rows
    intersected=joined_roads_in_aoi[joined_roads_in_aoi['OBJECTID_2'].notna()].copy()
    intersected_not_itself=intersected[intersected['OBJECTID_1']!=intersected['OBJECTID_2']].copy()
    intersected_roads=intersected_not_itself.drop_duplicates(subset=['OBJECTID_1','OBJECTID_2'])

    intersected_roads.reset_index(inplace=True, drop=True)

    ### Get rid of duplicates not on the same row
    to_drop=[]
    for idx in tqdm(intersected_roads.index, total=intersected_roads.shape[0],
                desc='-- Ereasing duplicates from spatial join'):
        ir1_objid=intersected_roads.loc[idx,'OBJECTID_1']
        ir2_objid=intersected_roads.loc[idx,'OBJECTID_2']
        
        for ss_idx in intersected_roads[intersected_roads['OBJECTID_1']==ir2_objid].index:
            
            if ir1_objid==intersected_roads.loc[ss_idx,'OBJECTID_2'] and idx<ss_idx:
                to_drop.append(ss_idx)

    intersected_roads.drop(to_drop,inplace=True)

    corr_overlap2=corr_overlap1.copy()

    ### from https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
    for idx in tqdm(intersected_roads.index, total=intersected_roads.shape[0],
                    desc='-- Suppressing overlap between equivalent roads'):
        
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
    print('-- Excluding roads under forest canopy ...')

    fct_misc.test_crs(corr_overlap2.crs, forests.crs)

    forests['buffered_geom']=forests.buffer(3)
    forests.drop(columns=['geometry'], inplace=True)
    forests.rename(columns={'buffered_geom':'geometry'}, inplace=True)

    non_forest_roads=corr_overlap2.copy()
    non_forest_roads=non_forest_roads.overlay(forests[['UUID','geometry']],how='difference')

    non_forest_roads.drop(columns=['UUID','GDB-Code','id'],inplace=True)
    non_forest_roads.rename(columns={'Width':'road_width'}, inplace=True)

    print('Done determining the surface of the roads from lines!')


if DETERMINE_RESTRICTED_AOI:
    print('Determing the restricted AOI around the considered roads...')

    print('-- Calculating buffer...')
    width=(roads_parameters['Width'].max()+1)/2

    buffered_roads_aoi=joined_roads_in_aoi.copy()
    buffered_roads_aoi['buffered_geom']=buffered_roads_aoi.buffer(width)
    
    buffered_roads_aoi.drop(columns=['geometry'],inplace=True)
    buffered_roads_aoi.rename(columns={'buffered_geom':'geometry'},inplace=True)

    AOI_roads=buffered_roads_aoi.unary_union

    print('-- Excluding parts under forest canopy...')
    geom={'geometry':[x for x in AOI_roads.geoms]}
    AOI_roads_no_forest=gpd.GeoDataFrame(geom, crs=roads.crs)

    fct_misc.test_crs(AOI_roads_no_forest.crs, forests.crs)
        
    AOI_roads_no_forest=AOI_roads_no_forest.overlay(forests[['UUID','geometry']],how='difference')

    print('Done determining the restricted AOI!')


if GENERATE_TILES_INFO:
    print('Determination of the information for the tiles to consider...')

    if not DETERMINE_RESTRICTED_AOI:
        AOI_roads_no_forest = gpd.read_file(os.path.join(OUTPUT_DIR, 'shapefiles_gpkg/restricted_AOI.shp'))
    
    bboxes_extent_4326=AOI_roads_no_forest.to_crs(epsg=4326).unary_union.bounds

    # cf. https://developmentseed.org/morecantile/usage/
    tms = morecantile.tms.get("WebMercatorQuad")    # epsg:3857

    print('-- Generating the tiles...')
    epsg3857_tiles_gdf = gpd.GeoDataFrame.from_features([tms.feature(x, projected=True) for x in tqdm(tms.tiles(*bboxes_extent_4326, zooms=[ZOOM_LEVEL]))])
    epsg3857_tiles_gdf.set_crs(epsg=3857, inplace=True)

    AOI_roads_no_forest_3857=AOI_roads_no_forest.to_crs(epsg=3857)
    AOI_roads_no_forest_3857.rename(columns={'FID': 'id_aoi'},inplace=True)

    print('-- Checking for intersections with the restricted area of interest...')
    fct_misc.test_crs(tms.crs, AOI_roads_no_forest_3857.crs)

    tiles_in_restricted_aoi=gpd.sjoin(epsg3857_tiles_gdf, AOI_roads_no_forest_3857, how='inner')

    print('-- Setting a formatted id...')
    tiles_in_restricted_aoi.drop_duplicates('geometry', inplace=True)
    tiles_in_restricted_aoi.drop(columns=['grid_name', 'grid_crs', 'index_right'], inplace=True)
    tiles_in_restricted_aoi.reset_index(drop=True, inplace=True)

    xyz=[]
    for idx in tiles_in_restricted_aoi.index:
        xyz.append([re.sub('[^0-9]','',coor) for coor in tiles_in_restricted_aoi.loc[idx,'title'].split(',')])

    tiles_in_restricted_aoi['id'] = [x+', '+y+', '+z for x, y, z in xyz]

    print('Done determining the tiles!')

if GENERATE_LABELS:
    print('Generating the labels for the object detector...')

    if not DETERMINE_ROAD_SURFACES:
        non_forest_roads=gpd.read_file(os.path.join(path_shp_gpkg, 'roads_polygons.shp'))

    if not GENERATE_TILES_INFO:
        tiles_in_restricted_aoi=gpd.read_file(os.path.join(path_json, 'tiles_aoi.geojson'))
    
    non_forest_roads=fct_misc.test_valid_geom(non_forest_roads, gdf_obj_name='the roads')

    labels_gdf = non_forest_roads.rename(columns={'BELAGSART': 'CATEGORY', 'road_width':'WIDTH'}).drop(columns=[
                        'DATUM_AEND', 'DATUM_ERST', 'ERSTELLUNG', 'ERSTELLU_1',
                        'REVISION_J', 'REVISION_M', 'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J',
                        'HERKUNFT_M', 'REVISION_Q', 'WANDERWEGE', 'VERKEHRSBE', 
                        'BEFAHRBARK', 'EROEFFNUNG', 'STUFE', 'RICHTUNGSG', 
                        'KREISEL', 'EIGENTUEME', 'VERKEHRS_1', 'NAME',
                        'TLM_STRASS', 'STRASSENNA', 'SHAPE_Leng',])
    labels_gdf['SUPERCATEGORY']='road'
    labels_gdf = labels_gdf.to_crs(epsg=4326)

    fct_misc.test_crs(labels_gdf.crs, tiles_in_restricted_aoi.crs)

    print('Done Generating the labels for the object detector...')

    labels_gdf=fct_misc.test_valid_geom(labels_gdf, correct=True, gdf_obj_name='the labels')

    GT_labels_gdf = gpd.sjoin(labels_gdf, tiles_in_restricted_aoi, how='inner', predicate='intersects')

    # the following two lines make sure that no object is counted more than once in case it intersects multiple tiles
    GT_labels_gdf = GT_labels_gdf[labels_gdf.columns]
    GT_labels_gdf.drop_duplicates(inplace=True)
    OTH_labels_gdf = labels_gdf[ ~labels_gdf.index.isin(GT_labels_gdf.index)]

    try:
        assert( len(labels_gdf) == len(GT_labels_gdf) + len(OTH_labels_gdf) ),\
            f"Something went wrong when splitting labels into Ground Truth Labels and Other Labels." +\
            f" Total no. of labels = {len(labels_gdf)}; no. of Ground Truth Labels = {len(GT_labels_gdf)}; no. of Other Labels = {len(OTH_labels_gdf)}"
    except Exception as e:
        print(e)
        sys.exit(1)

    # In the current case, OTH_labels_gdf should be empty

# Save results ------------------------------------------------------------------
print('Saving files...')

written_files=[]

if DETERMINE_ROAD_SURFACES:
    non_forest_roads.to_file(os.path.join(path_shp_gpkg, 'roads_polygons.shp'))
    written_files.append('shapefiles_gpkg/roads_polygons.shp')

if DETERMINE_RESTRICTED_AOI:
    AOI_roads_no_forest.to_file(os.path.join(path_shp_gpkg, 'restricted_AOI.shp'))
    written_files.append('shapefiles_gpkg/restricted_AOI.shp')

if GENERATE_TILES_INFO:
    # Save in json format for the (soon-to-be) "old" version of generate_tilesets.py
    # geojson only supports epsg:4326
    tiles_4326=tiles_in_restricted_aoi.to_crs(epsg=4326)
    tiles_4326.to_file(os.path.join(path_json, 'tiles_aoi.geojson'), driver='GeoJSON')

    # Save in gpkg for the (soon-to-be) new version of generate_tilesets.py
    layername="epsg3857_z" + str(ZOOM_LEVEL) + "_tiles"
    tiles_in_restricted_aoi.to_file(os.path.join(path_shp_gpkg, 'epsg3857_tiles.gpkg'), driver='GPKG',layer=layername)

    written_files.append('json/tiles_aoi.geojson')
    written_files.append(layername + ' in the geopackage shapefiles_gpkg/epsg3857_tiles.gpkg')

if GENERATE_LABELS:
    GT_labels_gdf.to_file(os.path.join(path_json, f'ground_truth_labels.geojson'), driver='GeoJSON')
    written_files.append('json/ground_truth_labels.geojson')

    if not OTH_labels_gdf.empty:
        OTH_labels_gdf.to_file(os.path.join(path_json, f'other_labels.geojson'), driver='GeoJSON')
        written_files.append('json/other_labels.geojson')

print('All done!')
print('Written files:')
for file in written_files:
    print(file)
