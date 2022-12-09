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
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['prepare_data_od.py']    #  [os.path.basename(__file__)]

# Task to do
DETERMINE_ROAD_SURFACES = cfg['tasks']['determine_roads_surfaces']
GENERATE_TILES_INFO=cfg['tasks']['generate_tiles_info']
GENERATE_LABELS=cfg['tasks']['generate_labels']

if not (DETERMINE_ROAD_SURFACES or GENERATE_TILES_INFO or GENERATE_LABELS) :
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
    NOT_ROAD=[12, 13, 14, 19, 22, 23]
    KUNSTBAUTE_TO_KEEP=[100, 200]
    BELAGSART_TO_KEEP=[100, 200]

    if 'ok_tiles' in cfg.keys():
        OK_TILES=OUTPUT_DIR+cfg['ok_tiles']
    else:
        OK_TILES=False

    if GENERATE_TILES_INFO or GENERATE_LABELS:
        ZOOM_LEVEL=cfg['zoom_level']

path_shp_gpkg=fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'shapefiles_gpkg'))
path_json=fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR,'json'))


# Information treatment -----------------------------------------------------------------------------------------

if DETERMINE_ROAD_SURFACES:
    print('Importing files...')

    ## Geodata
    roads=gpd.read_file(ROADS_IN)
    forests=gpd.read_file(FORESTS)

    ## Other informations
    roads_parameters=pd.read_excel(ROADS_PARAM)

    # Filter the roads to consider
    print('Filtering the considered roads...')
    
    roads_of_interest=roads[~roads['OBJEKTART'].isin(NOT_ROAD)]
    uncovered_roads=roads_of_interest[roads_of_interest['KUNSTBAUTE'].isin(KUNSTBAUTE_TO_KEEP)]

    if DEBUG_MODE:
        uncovered_roads=uncovered_roads[1:100]

    roads_parameters_filtered=roads_parameters[~roads_parameters['Width'].isna()].copy()
    roads_parameters_filtered.drop_duplicates(subset='GDB-Code',inplace=True)       # Keep first by default 

    uncovered_roads=uncovered_roads.merge(roads_parameters_filtered[['GDB-Code','Width']], how='inner',left_on='OBJEKTART',right_on='GDB-Code')

    uncovered_roads.drop(columns=[
                                'DATUM_AEND', 'DATUM_ERST', 'ERSTELLUNG', 'ERSTELLU_1', 'UUID',
                                'REVISION_J', 'REVISION_M', 'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J',
                                'HERKUNFT_M', 'REVISION_Q', 'WANDERWEGE', 'VERKEHRSBE', 
                                'BEFAHRBARK', 'EROEFFNUNG', 'STUFE', 'RICHTUNGSG', 
                                'KREISEL', 'EIGENTUEME', 'VERKEHRS_1', 'NAME',
                                'TLM_STRASS', 'STRASSENNA', 'SHAPE_Leng'], inplace=True)

    print('Determining the surface of the roads from lines...')

    # Buffer the roads
    print('-- Buffering the roads...')

    buffered_roads=uncovered_roads.copy()
    buffered_roads['geometry']=uncovered_roads.buffer(uncovered_roads['Width']/2, cap_style=2)

    ## Do not let roundabout parts make artifacts
    buff_geometries=[]
    for geom in buffered_roads['geometry'].values:
        if geom.geom_type == 'MultiPolygon':
            buff_geometries.append(max(geom.geoms, key=lambda a: a.area))
        else:
            buff_geometries.append(geom)
        
    buffered_roads['geometry'] = buff_geometries


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
    intersected_roads.loc[intersected_roads['OBJEKTART_1']==21,'OBJEKTART_1']=2.5

    intersect_other_width=intersected_roads[intersected_roads['OBJEKTART_1']<intersected_roads['OBJEKTART_2']].copy()

    intersect_other_width.sort_values(by=['OBJEKTART_1'],inplace=True)
    intersect_other_width.loc[intersect_other_width['OBJEKTART_1']==8.5,'OBJEKTART_1']=20
    intersect_other_width.loc[intersect_other_width['OBJEKTART_1']==2.5,'OBJEKTART_1']=21

    intersect_other_width.sort_values(by=['KUNSTBAUTE'], ascending=False, inplace=True, ignore_index=True)

    ### Suppress the overlapping intersection
    ### from https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
    corr_overlap = buffered_roads.copy()

    for idx in tqdm(intersect_other_width.index, total=intersect_other_width.shape[0],
                desc='-- Suppressing the overlap of roads with different width'):
        
        poly1_id = corr_overlap.index[corr_overlap['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_1']].values.astype(int)[0]
        poly2_id = corr_overlap.index[corr_overlap['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_2']].values.astype(int)[0]
        
        corr_overlap=fct_misc.polygons_diff_without_artifacts(corr_overlap, poly1_id, poly2_id, keep_everything=True)

    corr_overlap.drop(columns=['saved_geom'],inplace=True)
    corr_overlap.set_crs(epsg=2056, inplace=True)

    # Exclude the roads potentially under forest canopy
    print('-- Excluding roads under forest canopy ...')

    fct_misc.test_crs(corr_overlap.crs, forests.crs)

    forests['buffered_geom']=forests.buffer(3)
    forests.drop(columns=['geometry'], inplace=True)
    forests.rename(columns={'buffered_geom':'geometry'}, inplace=True)

    non_forest_roads=corr_overlap.copy()
    non_forest_roads=non_forest_roads.overlay(forests[['UUID','geometry']],how='difference')

    non_forest_roads.drop(columns=['GDB-Code'],inplace=True)
    non_forest_roads.rename(columns={'Width':'road_width'}, inplace=True)

    print('Done determining the surface of the roads from lines!')


if GENERATE_TILES_INFO or GENERATE_LABELS:

    if not DETERMINE_ROAD_SURFACES:
        print('Importing files...')
        non_forest_roads=gpd.read_file(os.path.join(path_shp_gpkg, 'roads_for_OD.shp'))
        roads_parameters=pd.read_excel(ROADS_PARAM)

if GENERATE_TILES_INFO:
    aoi=gpd.read_file(AOI)

    print('Determination of the information for the tiles to consider...')

    roads_parameters_filtered=roads_parameters[roads_parameters['to keep']=='yes'].copy()
    roads_parameters_filtered.drop_duplicates(subset='GDB-Code',inplace=True)       # Keep first by default 

    roads_of_interest=non_forest_roads.merge(roads_parameters_filtered[['GDB-Code']], how='right',left_on='OBJEKTART',right_on='GDB-Code')
    roads_of_interest=roads_of_interest[roads_of_interest['BELAGSART'].isin(BELAGSART_TO_KEEP)]


    aoi_geom=gpd.GeoDataFrame({'id': [0], 'geometry': [aoi['geometry'].unary_union]}, crs=2056)
    fct_misc.test_crs(roads_of_interest.crs, aoi_geom.crs)
    roi_in_aoi=roads_of_interest.overlay(aoi_geom, how='intersection')

    del roads_parameters, roads_parameters_filtered, roads_of_interest

    if DEBUG_MODE:
        roi_in_aoi=roi_in_aoi[1:100]

    roi_in_aoi=fct_misc.test_valid_geom(roi_in_aoi, gdf_obj_name='the roads')

    roi_in_aoi.drop(columns=['BELAGSART', 'road_width', 'OBJEKTART', 'OBJECTID', 'KUNSTBAUTE', 'GDB-Code'], inplace=True)
    
    bboxes_extent_4326=roi_in_aoi.to_crs(epsg=4326).unary_union.bounds

    # cf. https://developmentseed.org/morecantile/usage/
    tms = morecantile.tms.get("WebMercatorQuad")    # epsg:3857

    print('-- Generating the tiles...')
    epsg3857_tiles_gdf = gpd.GeoDataFrame.from_features([tms.feature(x, projected=True) for x in tqdm(tms.tiles(*bboxes_extent_4326, zooms=[ZOOM_LEVEL]))])
    epsg3857_tiles_gdf.set_crs(epsg=3857, inplace=True)

    roi_in_aoi_3857=roi_in_aoi.to_crs(epsg=3857)
    roi_in_aoi_3857.rename(columns={'FID': 'id_aoi'},inplace=True)

    print('-- Checking for intersections with the restricted area of interest...')
    fct_misc.test_crs(tms.crs, roi_in_aoi_3857.crs)

    tiles_in_restricted_aoi=gpd.sjoin(epsg3857_tiles_gdf, roi_in_aoi_3857, how='inner')

    print('-- Setting a formatted id...')
    tiles_in_restricted_aoi.drop_duplicates('geometry', inplace=True)
    tiles_in_restricted_aoi.drop(columns=['grid_name', 'grid_crs', 'index_right'], inplace=True)
    tiles_in_restricted_aoi.reset_index(drop=True, inplace=True)

    xyz=[]
    for idx in tiles_in_restricted_aoi.index:
        xyz.append([re.sub('[^0-9]','',coor) for coor in tiles_in_restricted_aoi.loc[idx,'title'].split(',')])

    tiles_in_restricted_aoi['id'] = ['('+ x +', '+y+', '+z + ')' for x, y, z in xyz]

    print('Done determining the tiles!')

if GENERATE_LABELS:
    print('Generating the labels for the object detector...')

    if not GENERATE_TILES_INFO:
        tiles_in_restricted_aoi_4326=gpd.read_file(os.path.join(path_json, 'tiles_aoi.geojson'))
    else:
        tiles_in_restricted_aoi_4326=tiles_in_restricted_aoi.to_crs(epsg=4326)

    if OK_TILES:
        if ZOOM_LEVEL==18:
            tiles_table=pd.read_excel(OK_TILES)
            tiles_table.replace('-','0.5', inplace=True)
            verified_tiles=tiles_table[~tiles_table['OK'].isna()].copy()
            verified_tiles=verified_tiles.astype({'OK': 'float'})

            ok_tiles=verified_tiles[verified_tiles['OK']>=0.5].copy()

            tiles_in_restricted_aoi_4326=tiles_in_restricted_aoi_4326.merge(ok_tiles, how='right', on='title')

            nbr_verified_tiles=verified_tiles.shape[0]
            per_unverified_tiles=round((tiles_table.shape[0]-nbr_verified_tiles)*100/tiles_table.shape[0],2)
            print(f"{per_unverified_tiles}% of the tiles are not verified yet.")

            per_rejected_tiles=round((nbr_verified_tiles-ok_tiles.shape[0])*100/nbr_verified_tiles,2)
            print(f"{per_rejected_tiles}% of the verified tiles were rejected")

            per_good_tiles=round((ok_tiles.shape[0]-ok_tiles[ok_tiles['OK']<0.75].shape[0])*100/nbr_verified_tiles,2)
            per_ok_tiles=round((ok_tiles.shape[0]-ok_tiles[ok_tiles['OK']>0.75].shape[0])*100/nbr_verified_tiles,2)
            print(f"{per_good_tiles}% of the verified tiles are good.")
            print(f"{per_ok_tiles}% of the verified tiles were are ok.")

        else:
            # TODO: generalize the tiles to the correct level or to the level 18 for comparison
            print('Ok tiles for this zoom not developped yet :(')

    # TODO: do the unary_union, but keep the road types separated 
    # roads_union=non_forest_roads.unary_union
    # labels_gdf_no_crs = gpd.GeoDataFrame({'id_labels':[x for x in range(len(roads_union.geoms))],
    #                                         'geometry':[geo for geo in roads_union.geoms]})
    # labels_gdf_no_crs['CATEGORY']='road'
    # labels_gdf_no_crs['SUPCATEGORY']='ground'
    # labels_gdf_2056=labels_gdf_no_crs.set_crs(epsg=2056)
    labels_gdf_2056=non_forest_roads.copy()
    labels_gdf_2056['CATEGORY']='road'
    labels_gdf_2056['SUPERCATEGORY']='ground'
    labels_gdf = labels_gdf_2056.to_crs(epsg=4326)
    labels_gdf=fct_misc.test_valid_geom(labels_gdf, correct=True, gdf_obj_name='the labels')

    fct_misc.test_crs(labels_gdf.crs, tiles_in_restricted_aoi_4326.crs)

    print('Labels on tiles...')
    tiles_union_geom=tiles_in_restricted_aoi_4326.unary_union
    tiles_union_df=gpd.GeoDataFrame({'id_temp':[x for x in range(len(tiles_union_geom.geoms))],
                                    'geometry':[geo for geo in tiles_union_geom.geoms]})
    tiles_union_df.set_crs(crs=tiles_in_restricted_aoi_4326.crs, inplace=True)
    labels_gdf=gpd.overlay(labels_gdf, tiles_union_df)
    labels_gdf.drop(columns=['id_temp'], inplace=True)

    GT_labels_gdf = gpd.sjoin(labels_gdf, tiles_in_restricted_aoi_4326, how='inner', predicate='intersects')

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

    print('Done Generating the labels for the object detector...')

    # In the current case, OTH_labels_gdf should be empty


# Save results ------------------------------------------------------------------
print('Saving files...')

written_files=[]

if DETERMINE_ROAD_SURFACES:
    non_forest_roads.to_file(os.path.join(path_shp_gpkg, 'roads_for_OD.shp'))
    written_files.append('shapefiles_gpkg/roads_for_OD.shp')

if GENERATE_TILES_INFO:
    # Save in json format for the (to-be) "old" version of generate_tilesets.py
    # geojson only supports epsg:4326
    tiles_4326=tiles_in_restricted_aoi.to_crs(epsg=4326)
    tiles_4326.to_file(os.path.join(path_json, 'tiles_aoi.geojson'), driver='GeoJSON')

    # Save in gpkg for the (soon-to-be) new version of generate_tilesets.py
    layername="epsg3857_z" + str(ZOOM_LEVEL) + "_tiles"
    tiles_in_restricted_aoi.to_file(os.path.join(path_shp_gpkg, 'epsg3857_tiles.gpkg'), driver='GPKG',layer=layername)

    written_files.append('json/tiles_aoi.geojson')
    written_files.append(layername + ' in the geopackage shapefiles_gpkg/epsg3857_tiles.gpkg')

if GENERATE_LABELS:
    GT_labels_gdf.to_file(os.path.join(path_json, 'ground_truth_labels.geojson'), driver='GeoJSON')
    written_files.append('json/ground_truth_labels.geojson')

    for road_type in BELAGSART_TO_KEEP:
        temp=GT_labels_gdf[GT_labels_gdf['BELAGSART']==road_type].copy()
        temp.to_file(os.path.join(path_json, f'ground_truth_labels_{road_type}.geojson'), driver='GeoJSON')
        written_files.append(f'json/ground_truth_labels_{road_type}.geojson')

    if not OTH_labels_gdf.empty:
        OTH_labels_gdf.to_file(os.path.join(path_json, f'other_labels.geojson'), driver='GeoJSON')
        written_files.append('json/other_labels.geojson')
        for road_type in BELAGSART_TO_KEEP:
            temp=OTH_labels_gdf[OTH_labels_gdf['BELAGSART']==road_type].copy()
            temp.to_file(os.path.join(path_json, f'other_labels_{road_type}.geojson'), driver='GeoJSON')
            written_files.append(f'json/other_labels_{road_type}.geojson')

print('All done!')
print('Written files:')
for file in written_files:
    print(file)