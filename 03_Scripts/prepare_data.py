from ntpath import join
import geopandas as gpd
import pandas as pd
from itertools import combinations
import math
import os, sys
import argparse
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
DETERMINE_TILES=cfg['tasks']['determine_tiles']

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
    
    if DETERMINE_TILES:
        TILES_AOI = OUTPUT_DIR + OUTPUT['output_files']['tiles_aoi']


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

def test_crs(crs1,crs2 = "EPSG:2056"):
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

    ## Get the features that intersect with a different class of roads
    print('Removing overlap between roads of different classes...')

    intersections=gpd.overlay(buffered_roads[['OBJECTID','geometry','OBJEKTART']],buffered_roads,how='intersection')

    intersect_others_dupli=intersections[intersections['OBJECTID_1']!=intersections['OBJECTID_2']].copy()
    road_diff_width=intersect_others_dupli[intersect_others_dupli['OBJEKTART_1']!=intersect_others_dupli['OBJEKTART_2']].copy()
    intersect_others=road_diff_width.drop_duplicates(subset=['OBJECTID_1'])

    id_to_test=intersect_others['OBJECTID_1'].tolist()

    ## Sort the dataframe by road size
    ### TO DO : automatize this part for when different object classes are kept
    if 20 in buffered_roads['OBJEKTART']:
        buffered_roads.loc[buffered_roads['OBJEKTART']==20,'OBJEKTART']=8.5
        buffered_roads.sort_values(by=['OBJEKTART'],inplace=True)
        buffered_roads.loc[buffered_roads['OBJEKTART']==8.5,'OBJEKTART']=20

    ## Remove the intersections between roads of different classes
    print('--- Comparing roads for intersections to remove...')

    poly=buffered_roads.loc[buffered_roads['OBJECTID'].isin(id_to_test),['OBJECTID','geometry','OBJEKTART']].copy()

    ### from https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
    iteration=0
    nbr_tot_iter=math.comb(poly.shape[0],2)
    for p1_idx, p2_idx in combinations(poly.index,2):
        
        if poly.loc[p1_idx,'geometry'].intersects(poly.loc[p2_idx,'geometry']) and poly.at[p1_idx,'OBJEKTART']!=poly.at[p2_idx,'OBJEKTART']:
            
            poly=polygons_diff_without_artifacts(poly,p1_idx,p2_idx)
            
        iteration += 1
        if iteration%1000000==0:
            percentage=iteration/nbr_tot_iter*100
            print(f'{round(percentage)}% done')

    print('100% done!')

    poly.rename(columns={'geometry':'geometry_cropped'}, inplace=True)
    corr_overlap1=buffered_roads.merge(poly,how='left',on='OBJECTID',suffixes=('_org','_cropped'))

    ### Change the corrected geometry
    print('--- Applying changes in the geometry...')

    geom=[]
    for idx in corr_overlap1.index:
        
        if not pd.isnull(corr_overlap1.at[idx,'geometry_cropped']):
            geom.append(corr_overlap1.at[idx,'geometry_cropped'])
        else:
            geom.append(corr_overlap1.at[idx,'geometry'])

    corr_overlap1.drop(columns={'geometry'},inplace=True)
    corr_overlap1['geometry']=geom
    corr_overlap1.set_crs(buffered_roads.crs,inplace=True)
    corr_overlap1.drop(columns=['OBJEKTART_cropped','geometry_cropped'],inplace=True)

    ## Remove overlapping area between roads of the same class
    print('Removing overlap between roads of the same classe...')

    road_same_width=intersect_others_dupli[intersect_others_dupli['OBJEKTART_1']==intersect_others_dupli['OBJEKTART_2']].copy()
    intersect_others=road_same_width.drop_duplicates(subset=['OBJECTID_1'])

    df=corr_overlap1.rename(columns={'OBJEKTART_org':'OBJEKTART'},errors='raise')
    id_to_test=intersect_others['OBJECTID_1'].tolist()
    poly=df.loc[df['OBJECTID'].isin(id_to_test),['OBJECTID','geometry','OBJEKTART']].copy()

    ### from https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
    print('--- Comparing roads for intersections to remove...')

    iteration=0
    nbr_iter_tot=math.comb(poly.shape[0],2)
    for p1_idx, p2_idx in combinations(poly.index,2):
        
        if poly.loc[p1_idx,'geometry'].intersects(poly.loc[p2_idx,'geometry']) and poly.at[p1_idx,'OBJEKTART']==poly.at[p2_idx,'OBJEKTART']:
            
            # Store intermediary results back to poly
            diff=poly.loc[p2_idx,'geometry']-poly.loc[p1_idx,'geometry']

            if diff.geom_type == 'Polygon':
                temp= poly.loc[p2_idx,'geometry']-poly.loc[p1_idx,'geometry']

            elif diff.geom_type == 'MultiPolygon':
                # if a multipolygone is created, only keep the largest part to avoid the following error: https://github.com/geopandas/geopandas/issues/992
                temp = max((poly.loc[p2_idx,'geometry']-poly.loc[p1_idx,'geometry']).geoms, key=lambda a: a.area)
                
            poly=polygons_diff_without_artifacts(poly,p2_idx,p1_idx)
        
            poly.loc[p2_idx,'geometry']=temp
            
        iteration += 1
        if iteration%1000000==0:
            percentage=iteration/nbr_iter_tot*100
            print(f'{round(percentage,1)}% done')

    print('100% done!')        
            
    poly.rename(columns={'geometry':'geometry_cropped'}, inplace=True)
    corr_overlap2=df.merge(poly,how='left',on='OBJECTID',suffixes=('_org','_cropped'))

    ### Change the corrected geometry
    print('--- Applying changes in the geometry...')

    geom=[]
    for idx in corr_overlap2.index:
        if not pd.isnull(corr_overlap2.at[idx,'geometry_cropped']):
            geom.append(corr_overlap2.at[idx,'geometry_cropped'])
        else:
            geom.append(corr_overlap2.at[idx,'geometry'])

    corr_overlap2.drop(columns={'geometry'},inplace=True)
    corr_overlap2['geometry']=geom
    corr_overlap2.set_crs(buffered_roads.crs,inplace=True)
    corr_overlap2.drop(columns=['OBJEKTART_cropped','geometry_cropped'],inplace=True)

    ## A try for a faster method
    '''test=corr_overlap1.rename(columns={'OBJEKTART_org':'OBJEKTART'},errors='raise')
    test['saved_geom']=test.geometry

    joined_test=gpd.sjoin(test,test[['OBJECTID','saved_geom','geometry']],how='left', lsuffix='1', rsuffix='2')
    intersected=joined_test[joined_test['OBJECTID_1']!=joined_test['OBJECTID_2']].copy()
    intersected_no_dupl=intersected.drop_duplicates(subset=['OBJECTID_1','OBJECTID_2'])
    print(intersected_no_dupl.shape)

    intersected_no_dupl['new_geom']=intersected_no_dupl['geometry'].difference(intersected_no_dupl['saved_geom_2'])
    intersected_no_dupl.drop(columns={'geometry'},inplace=True)
    intersected_no_dupl['geometry']=intersected_no_dupl['new_geom']

    intersected_no_dupl.drop(columns={'new_geom','saved_geom_1','index_2','OBJECTID_2','saved_geom_2'},inplace=True)'''

   ## INTERROGATION: should we remove the smallest polygons (risk of shadows or obsacles to important)? Or at least make sure they do not end up in 
   # the training dataset? Supressing them and just deducting their surface from their neighbours could be feseable, but it may be because
   # the dataset is very unbalanced.  

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


if DETERMINE_TILES:
    print('Downloading tiles from map.geo.admin WMS')

    if not DETERMINE_RESTRICTED_AOI:
        AOI_roads_no_forest = gpd.read_file(OUTPUT_DIR +  OUTPUT['output_files']['restricted_AOI'])
    
    test_crs(tiles_swissimages.crs, AOI_roads_no_forest.crs)

    tiles_in_restricted_aoi=gpd.sjoin(tiles_swissimages,AOI_roads_no_forest, how='inner')

    # We want the images from 2018
    tiles_in_restricted_aoi['datenstand']=2018

    tiles_in_restricted_aoi.drop(columns=['index_right', 'FID'], inplace=True)
    tiles_in_restricted_aoi.reset_index(inplace=True)

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

if DETERMINE_TILES:
    tiles_in_restricted_aoi.to_file(TILES_AOI)
    written_files.append(TILES_AOI)

print('All done!')
print(f'Written files: {written_files}')
