import os, sys
import yaml
import logging, logging.config
import time

import pandas as pd
import geopandas as gpd

from shapely.affinity import scale

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

# Define functions ------------------------------------

def get_corresponding_class(row, labels_id):
    'Get the class in words from the class ids out of the object detector with the method apply.'

    if row['pred_class']==0:
        return labels_id.loc[labels_id['id']==0, 'name'].item()
    elif row['pred_class']==1:
        return labels_id.loc[labels_id['id']==1, 'name'].item()
    else:
        logger.error(f"Unexpected class: {row['pred_class']}")
        sys.exit(1)

def determine_category(row):
    'Get the class in words from the codes out of the swissTLM3D with the method apply.'

    if row['BELAGSART']==100:
        return 'artificial'
    if row['BELAGSART']==200:
        return 'natural'
    else:
        logger.error(f"Unexpected class: {row['BELAGSART']}")
        sys.exit(1)

def get_roads_in_quarries(quarries, roads):
    '''
    Create a dataframe with the roads in a quarry based on a spatial join and the buffered quarries.

    - quarries: geodataframe of the quarries
    - roads: geodataframe of the roads
    return: a geodataframe of the roads in the quarries and another one with the other roads.
    '''

    buffered_quarries=quarries.copy()
    buffered_quarries['geometry']=buffered_quarries.buffer(5)
    buffered_quarries_4326=buffered_quarries.to_crs(epsg=4326)

    fct_misc.test_crs(roads.crs, buffered_quarries_4326.crs)

    roads_in_quarries=gpd.sjoin(roads, buffered_quarries_4326, predicate='within')
    roads_not_in_quarries=roads[~roads['OBJECTID'].isin(
                                    roads_in_quarries['OBJECTID'].unique().tolist())].reset_index(drop=True)

    return roads_in_quarries, roads_not_in_quarries

def clip_labels(labels_gdf, tiles_gdf, fact=0.99):
    '''
    Clip the labels to the tiles
    Copied from the misc functions of the object detector 
    cf. https://github.com/swiss-territorial-data-lab/object-detector/blob/master/helpers/misc.py

    - labels_gdf: geodataframe with the labels
    - tiles_gdf: geodataframe of the tiles
    - fact: factor to scale the tiles before clipping
    return: a geodataframe with the labels clipped to the tiles
    '''

    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']
        
    assert(labels_gdf.crs == tiles_gdf.crs)
    
    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate='intersects')
    
    def clip_row(row, fact=fact):
        
        old_geo = row.geometry
        scaled_tile_geo = scale(row.tile_geometry, xfact=fact, yfact=fact)
        new_geo = old_geo.intersection(scaled_tile_geo)
        row['geometry'] = new_geo

        return row

    clipped_labels_gdf = labels_tiles_sjoined_gdf.apply(lambda row: clip_row(row, fact), axis=1)
    clipped_labels_gdf.crs = labels_gdf.crs

    clipped_labels_gdf.drop(columns=['tile_geometry', 'index_right'], inplace=True)
    clipped_labels_gdf.rename(columns={'id': 'tile_id'}, inplace=True)

    return clipped_labels_gdf

def get_weighted_scores(ground_truth, predictions):
    '''
    Get the areas of intersection between the predictions and the labels and use them to weight the confidence score
    based on the percentage between the intersection area and the label area.

    - ground_truth: labels as a geodataframe
    - prediction: predictions as geodataframe
    return: a geodataframe with the intersection between predictions and labels with the weighted score as an attribute.
    '''

    ground_truth['area_label']=ground_truth.area

    fct_misc.test_crs(ground_truth.crs, predictions.crs)
    all_intersections=gpd.overlay(ground_truth, predictions, how='intersection')

    all_predicted_roads=all_intersections[(~all_intersections['BELAGSART'].isna()) &
                                                (~all_intersections['score'].isna())].copy()
    all_predicted_roads['joined_area']=all_predicted_roads.area
    all_predicted_roads['area_pred_in_label']=round(all_predicted_roads['joined_area']/all_predicted_roads['area_label'], 2)
    all_predicted_roads['weighted_score']=all_predicted_roads['area_pred_in_label']*all_predicted_roads['score']

    predicted_roads=all_predicted_roads[all_predicted_roads.area_pred_in_label > 0.05].copy()

    return predicted_roads

def determine_detected_class(predictions, roads, threshold=0):
    '''
    Determine the detected class for the road surface by combining the multiple detections.

    - predictions: dataframe of the predicted road surface
    - roads: dataframe of the road polygons
    - threshold: threshold for the confidence score
    return: a dataframe with predicted surface type and the combined artificial and natural confidence score, as well as their difference.
        If available, return the actual suface type from the gt.
    '''

    final_type={'road_id':[], 'cover_type':[], 'nat_score':[], 'art_score':[], 'diff_score':[]}
    valid_predictions=predictions[predictions['score']>=threshold]
    detected_roads_id=valid_predictions['OBJECTID'].unique().tolist()

    for road_id in roads['OBJECTID'].unique().tolist():

        if road_id not in detected_roads_id:
            final_type['road_id'].append(road_id)
            final_type['cover_type'].append('undetected')
            final_type['nat_score'].append(0)
            final_type['art_score'].append(0)
            final_type['diff_score'].append(0)
            continue

        intersecting_predictions=valid_predictions[valid_predictions['OBJECTID']==road_id].copy()

        groups=intersecting_predictions.groupby(['pred_class_name']).sum(numeric_only=True)
        if 'natural' in groups.index:
            if groups.loc['natural', 'weighted_score']==0:
                natural_index=0
            else:
                natural_index=groups.loc['natural', 'weighted_score']/groups.loc['natural', 'area_pred_in_label']
        else:
            natural_index=0
        if 'artificial' in groups.index:
            if groups.loc['artificial', 'weighted_score']==0:
                artificial_index=0
            else:
                artificial_index=groups.loc['artificial', 'weighted_score']/groups.loc['artificial', 'area_pred_in_label']
        else:
            artificial_index=0

        if artificial_index==natural_index:
            final_type['road_id'].append(road_id)
            final_type['cover_type'].append('undetermined')
            final_type['diff_score'].append(0)
        elif artificial_index > natural_index:
            final_type['road_id'].append(road_id)
            final_type['cover_type'].append('artificial')
            final_type['diff_score'].append(abs(artificial_index-natural_index))
        elif artificial_index < natural_index:
            final_type['road_id'].append(road_id)
            final_type['cover_type'].append('natural')
            final_type['diff_score'].append(abs(artificial_index-natural_index))

        final_type['art_score'].append(round(artificial_index,3))
        final_type['nat_score'].append(round(natural_index, 3))

    final_type_df=pd.DataFrame(final_type)

    columns_to_keep=['OBJECTID', 'geometry']
    if 'gt_type' in roads.columns:
        columns_to_keep.extend(['CATEGORY', 'gt_type'])

    comparison_df=gpd.GeoDataFrame(final_type_df.merge(roads[columns_to_keep],
                                how='inner', left_on='road_id', right_on='OBJECTID'))

    return comparison_df

if __name__ == "__main__":

    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('root')

    tic = time.time()
    logger.info('Starting...')

    logger.info(f"Using config.yaml as config file.")
    with open('config/config_obj_detec.yaml') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)['determine_class.py']

    # Define constants ------------------------------------

    INITIAL_FOLDER=cfg['initial_folder']
    PROCESSED_FOLDER=cfg['processed_folder']
    FINAL_FOLDER=cfg['final_folder']

    THRESHOLD=cfg['threshold']

    ROAD_PARAMETERS=os.path.join(INITIAL_FOLDER, cfg['inputs']['road_param'])

    ROADS=os.path.join(PROCESSED_FOLDER, cfg['inputs']['roads'])

    PREDICTIONS=os.path.join(PROCESSED_FOLDER, cfg['inputs']['predictions'])
    TILES=os.path.join(PROCESSED_FOLDER, cfg['inputs']['tiles'])
    LABELS_ID=os.path.join(PROCESSED_FOLDER, cfg['inputs']['labels_id'])

    QUARRIES=os.path.join(INITIAL_FOLDER, cfg['inputs']['quarries'])

    shp_gpkg_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'shp_gpkg'))

    written_files=[]

    # Importing files ----------------------------------
    logger.info('Importing files...')

    road_parameters=pd.read_excel(ROAD_PARAMETERS)

    initial_road_polygons=gpd.read_file(ROADS)

    labels_id=pd.read_json(LABELS_ID, orient='index')
    logger.info('Possible classes:')
    for combination in labels_id.itertuples():
            print(f"- {combination.id}: {combination.name}, {combination.supercategory}")
    CLASSES=labels_id['name'].unique().tolist()

    predictions=gpd.read_file(PREDICTIONS)
    predictions['pred_class_name']=predictions.apply(lambda row: get_corresponding_class(row, labels_id), axis=1)
    predictions.drop(columns=['pred_class'], inplace=True)

    tiles=gpd.read_file(TILES)

    quarries=gpd.read_file(QUARRIES)

    # Information treatment ----------------------------

    logger.info('Filtering the GT for the roads of interest...')
    filtered_road_parameters=road_parameters[road_parameters['to keep']=='yes'].copy()
    filtered_road_polys=initial_road_polygons.merge(filtered_road_parameters[['GDB-Code','Width']], 
                                            how='inner',left_on='OBJEKTART',right_on='GDB-Code')

    # Roads in quarries are always naturals
    logger.info('-- Roads in quarries are always naturals...')

    roads_in_quarries, filtered_road_polys = get_roads_in_quarries(quarries, filtered_road_polys)
    filepath=os.path.join(shp_gpkg_folder, 'roads_in_quarries.shp')
    roads_in_quarries.to_file(filepath)
    written_files.append(os.path.join(filepath))

    logger.info('Limiting the labels to the visible area of labels and predictions...')

    visible_road_polys=clip_labels(filtered_road_polys, tiles[['title', 'id', 'geometry']])

    logger.info('Getting the intersecting area between predictions and labels...')

    visible_road_polys_2056=visible_road_polys.to_crs(epsg=2056)
    predictions_2056=predictions.to_crs(epsg=2056)

    predicted_roads=get_weighted_scores(visible_road_polys_2056, predictions_2056)

    del visible_road_polys_2056, initial_road_polygons, predictions_2056

    final_roads=determine_detected_class(predicted_roads, filtered_road_polys, THRESHOLD)
    
    filepath=os.path.join(shp_gpkg_folder, 'types_from_detections.shp')
    final_roads.to_file(filepath)
    written_files.append(filepath)