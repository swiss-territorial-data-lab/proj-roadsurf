import os
import sys
import argparse
import time
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger = fct_misc.format_logger(logger)

tic = time.time()
logger.info('Starting...')


parser = argparse.ArgumentParser(description="This script prepare the data for the statistical procedure.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


# Task to do
DETERMINE_ROAD_SURFACES = cfg['tasks']['determine_roads_surfaces']
DETERMINE_RESTRICTED_AOI = cfg['tasks']['determine_restricted_AOI']

if not (DETERMINE_ROAD_SURFACES or DETERMINE_RESTRICTED_AOI) :
    logger.info('Nothing to do. Exiting!')
    sys.exit(0)
else:

    INPUT = cfg['input']
    INPUT_DIR =INPUT['input_folder']

    ROADS_IN = os.path.join(INPUT_DIR, INPUT['input_files']['roads'])
    ROADS_PARAM = os.path.join(INPUT_DIR, INPUT['input_files']['roads_param'])
    FORESTS = os.path.join(INPUT_DIR, INPUT['input_files']['forests'])
    AOI = os.path.join(INPUT_DIR, INPUT['input_files']['aoi'])

    OUTPUT_DIR = cfg['output_folder']
    
    DEBUG_MODE=cfg['debug_mode']
    KUNSTBAUTE_TO_KEEP=[100, 200]
    BELAGSART_TO_KEEP=[100, 200]


# Information treatment -----------------------------------------------------------------------------------------

if DETERMINE_ROAD_SURFACES or DETERMINE_RESTRICTED_AOI:

    logger.info('Importing files...')

    ## Geodata
    roads=gpd.read_file(ROADS_IN)
    forests=gpd.read_file(FORESTS)
    aoi=gpd.read_file(AOI)

    ## Other informations
    roads_parameters=pd.read_excel(ROADS_PARAM)

    logger.info('Filtering the considered roads...')
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
    logger.info('Determining the surface of the roads from lines...')

    joined_roads_in_aoi['road_len']=round(joined_roads_in_aoi.length,3)

    logger.info('-- Buffering the roads...')

    buffered_roads=joined_roads_in_aoi.copy()
    buffered_roads['buffered_geom']=buffered_roads.buffer(joined_roads_in_aoi['Width']/2, cap_style=2)

    buffered_roads.drop(columns=['geometry'],inplace=True)
    buffered_roads.rename(columns={'buffered_geom':'geometry'},inplace=True)

    # Erease artifact polygons produced by roundabouts
    for idx in buffered_roads.index:
        geom=buffered_roads.loc[idx,'geometry']
        if geom.geom_type == 'MultiPolygon':
            buffered_roads.loc[idx,'geometry'] = max(buffered_roads.loc[idx,'geometry'].geoms, key=lambda a: a.area)

    # Erase overlapping zones of roads buffer
    logger.info('-- Comparing roads for intersections to remove...')

    logger.info('----- Removing overlap between roads of different classes...')

    buffered_roads['saved_geom']=buffered_roads.geometry
    joined_roads_in_aoi=gpd.sjoin(buffered_roads,buffered_roads[['OBJECTID','OBJEKTART','saved_geom','geometry']],
                                how='left', lsuffix='1', rsuffix='2')

    ## Drop excessive rows
    intersected=joined_roads_in_aoi[joined_roads_in_aoi['OBJECTID_2'].notna()].copy()
    intersected_not_itself=intersected[intersected['OBJECTID_1']!=intersected['OBJECTID_2']].copy()
    intersected_roads=intersected_not_itself.drop_duplicates(subset=['OBJECTID_1','OBJECTID_2'])

    intersected_roads.reset_index(inplace=True, drop=True)

    ## Sort the roads so that the widest ones come first
    intersected_roads.loc[intersected_roads['OBJEKTART_1']==20,'OBJEKTART_1']=8.5

    intersect_other_width=intersected_roads[intersected_roads['OBJEKTART_1']<intersected_roads['OBJEKTART_2']].copy()

    intersect_other_width.sort_values(by=['OBJEKTART_1'],inplace=True)
    intersect_other_width.loc[intersect_other_width['OBJEKTART_1']==8.5,'OBJEKTART_1']=20

    intersect_other_width.reset_index(inplace=True, drop=True)

    # cf. https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
    corr_overlap1 = buffered_roads.copy()
    for idx in tqdm(intersect_other_width.index, total=intersect_other_width.shape[0],
                desc='-- Suppressing the overlap of roads with different width'):
        
        poly1_id = corr_overlap1.index[
                corr_overlap1['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_1']
            ].values.astype(int)[0]
        poly2_id = corr_overlap1.index[
                corr_overlap1['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_2']
            ].values.astype(int)[0]
        
        corr_overlap1=fct_misc.polygons_diff_without_artifacts(corr_overlap1,poly1_id,poly2_id)

    corr_overlap1.drop(columns=['saved_geom'],inplace=True)
        
    logger.info('----- Removing overlap between roads of the same class...')

    save_geom=corr_overlap1.copy()
    save_geom['saved_geom']=save_geom.geometry
    joined_roads_in_aoi=gpd.sjoin(save_geom,save_geom[['OBJECTID','saved_geom','geometry']],
                                how='left', lsuffix='1', rsuffix='2')

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

    # cf https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
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
            # if a multipolygone is created, only keep the largest part to avoid the following error: 
            # https://github.com/geopandas/geopandas/issues/992
            temp = max((geom2 - geom1).geoms, key=lambda a: a.area)

        corr_overlap2=fct_misc.polygons_diff_without_artifacts(corr_overlap2,poly2_id,poly1_id)

        corr_overlap2.loc[poly2_id,'geometry']=temp

    logger.info('-- Excluding roads under forest canopy ...')

    fct_misc.test_crs(corr_overlap2.crs, forests.crs)

    forests['buffered_geom']=forests.buffer(3)
    forests.drop(columns=['geometry'], inplace=True)
    forests.rename(columns={'buffered_geom':'geometry'}, inplace=True)

    non_forest_roads=corr_overlap2.copy()
    non_forest_roads=non_forest_roads.overlay(forests[['UUID','geometry']],how='difference')

    non_forest_roads.drop(columns=['UUID','GDB-Code','id'],inplace=True)
    non_forest_roads.rename(columns={'Width':'road_width'}, inplace=True)

    logger.info('Done determining the surface of the roads from lines!')


if DETERMINE_RESTRICTED_AOI:
    logger.info('Determing the restricted AOI around the considered roads...')

    logger.info('-- Calculating buffer...')
    width=(roads_parameters['Width'].max()+1)/2

    buffered_roads_aoi=joined_roads_in_aoi.copy()
    buffered_roads_aoi['buffered_geom']=buffered_roads_aoi.buffer(width)
    
    buffered_roads_aoi.drop(columns=['geometry'],inplace=True)
    buffered_roads_aoi.rename(columns={'buffered_geom':'geometry'},inplace=True)

    AOI_roads=buffered_roads_aoi.unary_union

    logger.info('-- Excluding parts under forest canopy...')
    geom={'geometry':[x for x in AOI_roads.geoms]}
    AOI_roads_no_forest=gpd.GeoDataFrame(geom, crs=roads.crs)

    fct_misc.test_crs(AOI_roads_no_forest.crs, forests.crs)
        
    AOI_roads_no_forest=AOI_roads_no_forest.overlay(forests[['UUID','geometry']],how='difference')

    logger.info('Done determining the restricted AOI!')


# Save results ------------------------------------------------------------------
logger.info('Saving files...')

written_files=[]

path_shp_gpkg=fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'shapefiles_gpkg'))

if DETERMINE_ROAD_SURFACES:
    filepath=os.path.join(path_shp_gpkg, 'roads_polygons_stats.shp')
    non_forest_roads.to_file(filepath)
    written_files.append(filepath)

if DETERMINE_RESTRICTED_AOI:
    filepath=os.path.join(path_shp_gpkg, 'restricted_AOI.shp')
    AOI_roads_no_forest.to_file(filepath)
    written_files.append(filepath)

logger.info('All done!')
logger.info('Written files:')
for file in written_files:
    logger.info(file)
