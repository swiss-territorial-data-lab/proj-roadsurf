import os, sys
import yaml, argparse
import logging, logging.config
import time

import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from shapely.affinity import scale

from tqdm import tqdm

sys.path.insert(0, 'scripts')
import fct_misc

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

tic = time.time()
logger.info('Starting...')

# parser = argparse.ArgumentParser(description="This script trains a predictive models.")
# parser.add_argument('config_file', type=str, help='a YAML config file')
# args = parser.parse_args()

# logger.info(f"Using {args.config_file} as config file.")
# with open(args.config_file) as fp:
#         cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

logger.info(f"Using config.yaml as config file.")
with open('config.yaml') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['final_metrics.py']


# Define constants ------------------------------------

INITIAL_FOLDER=cfg['initial_folder']
PROCESSED_FOLDER=cfg['processed_folder']
FINAL_FOLDER=cfg['final_folder']

ROAD_PARAMETERS=os.path.join(INITIAL_FOLDER, cfg['input']['road_param'])

GROUND_TRUTH=os.path.join(PROCESSED_FOLDER, cfg['input']['ground_truth'])
if 'other_labels' in cfg['input'].keys():
    OTHER_LABELS=os.path.join(PROCESSED_FOLDER, cfg['input']['other_labels'])
else:
    OTHER_LABELS=None

PREDICTIONS=cfg['input']['to_evaluate']
TILES=os.path.join(PROCESSED_FOLDER, cfg['input']['tiles'])
LABELS_ID=os.path.join(PROCESSED_FOLDER, cfg['input']['labels_id'])

shp_gpkg_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'shp_gpkg'))

written_files=[]

# Definition of functions ---------------------------

def determine_detected_class(predictions, ground_truth, threshold=0):
    '''
    Determine the detected class for the road surface by combining the multiple detections.

    - predictions: dataframe of the predicted road surface
    - ground_truth: dataframe of the ground truth
    - threshold: threshold for the confidence score
    return: a dataframe with the actual and predicted surface type and the combined artificial and natural confidence score,
        as well as their difference.
    '''

    final_type={'road_id':[], 'cover_type':[], 'nat_score':[], 'art_score':[], 'diff_score':[]}
    valid_predictions=predictions[predictions['score']>=threshold]
    detected_roads_id=valid_predictions['OBJECTID'].unique().tolist()

    for road_id in ground_truth['OBJECTID'].unique().tolist():

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

    comparison_df=gpd.GeoDataFrame(final_type_df.merge(ground_truth[['OBJECTID','geometry', 'CATEGORY', 'gt_type']],
                                    how='inner', left_on='road_id', right_on='OBJECTID'))

    return comparison_df


def get_balanced_accuracy(comparison_df, CLASSES):
    '''
    Get a dataframe with the GT, the predictions and the tags (TP, FP, FN)
    Calculate the per-class and global precision, recall and f1-score.

    - comparison_df: dataframe with the GT and the predictions
    - CLASSES: classes to search for

    return: a dataframe with the TP, FP, FN, precision and recall per class and a second one with the global metrics.
    '''

    metrics_dict={'cover_class':[], 'TP':[], 'FP':[], 'FN':[], 'Pk':[], 'Rk':[], 'count':[]}
    for cover_type in CLASSES:
        metrics_dict['cover_class'].append(cover_type)
        tp=comparison_df[(comparison_df['tag']=='TP') &
                        (comparison_df['CATEGORY']==cover_type)].shape[0]
        fp=comparison_df[(comparison_df['tag']=='wrong class') &
                        (comparison_df['cover_type']==cover_type)].shape[0]
        fn_class=comparison_df[(comparison_df['tag']=='wrong class') &
                        (comparison_df['CATEGORY']==cover_type)].shape[0]
        fn=comparison_df[(comparison_df['tag']=='FN') &
                        (comparison_df['CATEGORY']==cover_type)].shape[0]

        metrics_dict['TP'].append(tp)
        metrics_dict['FP'].append(fp)
        metrics_dict['FN'].append(fn+fn_class)

        if tp==0:
            pk=0
            rk=0
        else:
            pk=tp/(tp+fp)
            rk=tp/(tp+fn+fn_class)

        metrics_dict['Pk'].append(pk)
        metrics_dict['Rk'].append(rk)

        metrics_dict['count'].append(comparison_df[comparison_df['CATEGORY']==cover_type].shape[0])

    metrics_df=pd.DataFrame(metrics_dict)

    total_roads_by_type=metrics_df['count'].sum()

    weighted_precision=(metrics_df['Pk']*metrics_df['count']).sum()/total_roads_by_type
    weighted_recall=(metrics_df['Rk']*metrics_df['count']).sum()/total_roads_by_type

    if weighted_precision==0 and weighted_recall==0:
        weighted_f1_score=0
    else:
        weighted_f1_score=round(2*weighted_precision*weighted_recall/(weighted_precision + weighted_recall), 2)

    balanced_precision=metrics_df['Pk'].sum()/2
    balanced_recall=metrics_df['Rk'].sum()/2

    if balanced_precision==0 and balanced_recall==0:
        balanced_f1_score=0
    else:
        balanced_f1_score=round(2*balanced_precision*balanced_recall/(weighted_precision + balanced_recall), 2)

    global_metrics_df=pd.DataFrame({'Pw': [weighted_precision], 'Rw': [weighted_recall], 'f1w': [weighted_f1_score],
                                    'Pb': [balanced_precision], 'Rb': [balanced_recall], 'f1b': [balanced_f1_score]})

    return metrics_df, global_metrics_df

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
        # return 'unknown'

def get_tag(row):
        'Compare the class in the prediction and the GT and tag the row as TP, FP or FN with the method apply.'

        pred_class=row.cover_type
        gt_class=row.CATEGORY

        if pred_class=='undetermined' or pred_class=='undetected':
            return 'FN'
        elif pred_class==gt_class:
            return 'TP'
        elif pred_class!=gt_class:
             return 'wrong class'
        else:
            logger.error(f'Unexpected configuration: prediction class is {pred_class} and ground truth class is {gt_class}.')
            sys.exit(1)

def show_metrics(metrics_by_class, global_metrics):
    '''
    Print the by-class precision and recall and the global precision, recall and f1-score

    - metrics_by_class: The by-class metrics as given by the function get_balanced_accuracy()
    - global_metrics: The global metrics as given by the function get_balanced_accuracy()

    return: -
    '''

    for metric in metrics_by_class.itertuples():
        logger.info('%s %s',
            f"The {metric.cover_class} roads have a precision of {round(metric.Pk, 2)}",
            f"and a recall of {round(metric.Rk, 2)}")

    logger.info('%s %s %s',
        f"The final f1-score is {global_metrics.f1w[0]}", 
        f"with a precision of {round(global_metrics.Pw[0],2)} and a recall of",
        f"{round(global_metrics.Rw[0],2)}.")


# Importing files ----------------------------------
logger.info('Importing files...')

road_parameters=pd.read_excel(ROAD_PARAMETERS)

# ground_truth=gpd.read_file(GROUND_TRUTH, layer=LAYER)
ground_truth=gpd.read_file(GROUND_TRUTH)
ground_truth['gt_type']='gt'
if OTHER_LABELS:
    other_labels=gpd.read_file(OTHER_LABELS)
    other_labels['gt_type']='oth'
    ground_truth=pd.concat([ground_truth, other_labels], ignore_index=True)

labels_id=pd.read_json(LABELS_ID, orient='index')

predictions=gpd.GeoDataFrame()
for dataset_acronym in PREDICTIONS.keys():
    dataset=gpd.read_file(os.path.join(PROCESSED_FOLDER, PREDICTIONS[dataset_acronym]))
    dataset['dataset']=dataset_acronym
    predictions=pd.concat([predictions, dataset], ignore_index=True)
predictions['pred_class_name']=predictions.apply(lambda row: get_corresponding_class(row, labels_id), axis=1)
predictions.drop(columns=['pred_class'], inplace=True)

tiles=gpd.read_file(TILES)
considered_tiles=tiles[tiles['dataset'].isin(PREDICTIONS.keys())]
validation_tiles=tiles[tiles['dataset']=='val']

quarries=gpd.read_file(os.path.join(INITIAL_FOLDER, 'quarries/quarries.shp'))

del dataset, tiles

# Information treatment ----------------------------

logger.info('Possible classes:')
for combination in labels_id.itertuples():
        print(f"- {combination.id}: {combination.name}, {combination.supercategory}")
CLASSES=labels_id['name'].unique().tolist()

logger.info('Filtering the GT for the roads of interest...')
filtered_road_parameters=road_parameters[road_parameters['to keep']=='yes'].copy()
filtered_ground_truth=ground_truth.merge(filtered_road_parameters[['GDB-Code','Width']], 
                                        how='inner',left_on='OBJEKTART',right_on='GDB-Code')
filtered_ground_truth=filtered_ground_truth[filtered_ground_truth['BELAGSART']!=999997]

filtered_ground_truth['CATEGORY']=filtered_ground_truth.apply(lambda row: determine_category(row), axis=1)

# Roads in quarries are always naturals
logger.info('-- Roads in quarries are always naturals...')

buffered_quarries=quarries.copy()
buffered_quarries['geometry']=buffered_quarries.buffer(5)
buffered_quarries_4326=buffered_quarries.to_crs(epsg=4326)

fct_misc.test_crs(filtered_ground_truth.crs, buffered_quarries_4326.crs)

roads_in_quarries=gpd.sjoin(filtered_ground_truth, buffered_quarries_4326, predicate='within')
filepath=os.path.join(shp_gpkg_folder, 'roads_in_quarries.shp')
roads_in_quarries.to_file(filepath)
written_files.append(os.path.join(filepath))

filtered_ground_truth=filtered_ground_truth[~filtered_ground_truth['OBJECTID'].isin(
                                    roads_in_quarries['OBJECTID'].unique().tolist())] 

logger.info('Limiting the labels to the visible area of labels and predictions...')

tiles_union=considered_tiles['geometry'].unary_union
considered_zone=gpd.GeoDataFrame({'id_tiles_union': [i for i in range(len(tiles_union.geoms))],
                                'geometry': [geo for geo in tiles_union.geoms]},
                                crs=4326
                                )

visible_ground_truth=clip_labels(filtered_ground_truth, considered_tiles[['title', 'id', 'geometry']])

del considered_tiles, tiles_union, considered_zone, 

logger.info('Getting the intersecting area between predictions and labels...')

ground_truth_2056=visible_ground_truth.to_crs(epsg=2056)
ground_truth_2056['area_label']=ground_truth_2056.area

predictions_2056=predictions.to_crs(epsg=2056)

fct_misc.test_crs(ground_truth_2056.crs, predictions_2056.crs)
predicted_roads_2056=gpd.overlay(ground_truth_2056, predictions_2056, how='intersection')

predicted_roads_filtered=predicted_roads_2056[(~predicted_roads_2056['BELAGSART'].isna()) &
                                            (~predicted_roads_2056['score'].isna())].copy()
predicted_roads_filtered['joined_area']=predicted_roads_filtered.area
predicted_roads_filtered['area_pred_in_label']=round(predicted_roads_filtered['joined_area']/predicted_roads_filtered['area_label'], 2)
predicted_roads_filtered['weighted_score']=predicted_roads_filtered['area_pred_in_label']*predicted_roads_filtered['score']

del ground_truth, ground_truth_2056, filtered_ground_truth, predictions_2056

logger.info('Determining the best metrics for the predictions based on the validation dataset...')
val_predictions=predicted_roads_filtered[predicted_roads_filtered['dataset']=='val']
validation_ground_truth=visible_ground_truth[visible_ground_truth.geometry.intersects(validation_tiles.unary_union)]

all_global_metrics=pd.DataFrame()
all_metrics_by_class=pd.DataFrame()

thresholds=np.arange(0, 1., 0.05)
tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

for threshold in thresholds:
    tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

    val_comparison_df=determine_detected_class(val_predictions, validation_ground_truth, threshold)

    val_comparison_df['tag']=val_comparison_df.apply(lambda row: get_tag(row), axis=1)

    part_metrics_by_class, part_global_metrics = get_balanced_accuracy(val_comparison_df, CLASSES)

    part_metrics_by_class['threshold']=threshold
    part_global_metrics['threshold']=threshold

    all_metrics_by_class=pd.concat([all_metrics_by_class, part_metrics_by_class], ignore_index=True)
    all_global_metrics=pd.concat([all_global_metrics, part_global_metrics], ignore_index=True)

    if threshold==0:
        best_threshold=0
        max_f1=part_global_metrics.f1b[0]

    elif (part_global_metrics.f1b>max_f1)[0]:
        best_threshold=threshold
        max_f1=part_global_metrics.f1b[0]
        
        best_val_by_class_metrics=part_metrics_by_class
        best_val_global_metrics=part_global_metrics

        print('\n')
        logger.info(f"The best threshold for the f1-score is now {best_threshold}.")

    tqdm_log.update(1)

tqdm_log.close()

logger.info("Metrics for the validation dataset:")
show_metrics(best_val_by_class_metrics, best_val_global_metrics)

print('\n')
logger.info(f"For a threshold of {best_threshold}...")
comparison_df=determine_detected_class(predicted_roads_filtered, visible_ground_truth, best_threshold)

try:
    assert(comparison_df.shape[0]==visible_ground_truth.shape[0]), "There are too many or not enough labels in the final results"
except Exception as e:
    logger.error(e)
    sys.exit(1)

comparison_df['tag']=comparison_df.apply(lambda row: get_tag(row), axis=1)

best_metrics_by_class, best_global_metrics = get_balanced_accuracy(comparison_df, CLASSES)

best_comparison_df=comparison_df.copy()

show_metrics(best_metrics_by_class, best_global_metrics)

filepath=os.path.join(shp_gpkg_folder, 'types_from_detections.shp')
best_comparison_df.to_file(filepath)
written_files.append(filepath)

print('\n')
logger.info(f"If we were to keep all the predictions, the metrics would be...")
all_preds_comparison_df=determine_detected_class(predicted_roads_filtered, visible_ground_truth, 0)

all_preds_comparison_df['tag']=all_preds_comparison_df.apply(lambda row: get_tag(row), axis=1)

all_preds_metrics_by_class, all_preds_global_metrics = get_balanced_accuracy(all_preds_comparison_df, CLASSES)

show_metrics(all_preds_metrics_by_class, all_preds_global_metrics)

filepath=os.path.join(shp_gpkg_folder, 'types_from_all_detections.shp')
all_preds_comparison_df.to_file(filepath)
written_files.append(filepath)

if  'oth' in PREDICTIONS.keys():
    print('\n')
    logger.info('Metrics based on the trn, tst, val datasets...')

    not_oth_predictions=predicted_roads_filtered[predicted_roads_filtered['dataset'].isin(['trn', 'tst', 'val'])]
    ground_truth_from_gt=visible_ground_truth[visible_ground_truth['gt_type']=='gt']
    not_oth_comparison_df=determine_detected_class(not_oth_predictions, ground_truth_from_gt, best_threshold)

    not_oth_comparison_df['tag']=not_oth_comparison_df.apply(lambda row: get_tag(row), axis=1)

    not_oth_metrics_by_class, not_oth_global_metrics = get_balanced_accuracy(not_oth_comparison_df, CLASSES)

    show_metrics(not_oth_metrics_by_class, not_oth_global_metrics)

    print('\n')
    logger.info('Metrics based on the predictions of the oth dataset...')

    oth_predictions=predicted_roads_filtered[predicted_roads_filtered['dataset']=='oth']
    ground_truth_from_oth=visible_ground_truth[visible_ground_truth['gt_type']=='oth']
    oth_comparison_df=determine_detected_class(oth_predictions, ground_truth_from_oth, best_threshold)

    oth_comparison_df['tag']=oth_comparison_df.apply(lambda row: get_tag(row), axis=1)

    oth_metrics_by_class, oth_global_metrics = get_balanced_accuracy(oth_comparison_df,  CLASSES)

    show_metrics(oth_metrics_by_class, oth_global_metrics)

print('\n')
logger.info('-- Calculating the accuracy...')

per_right_roads=best_comparison_df[best_comparison_df['CATEGORY']==best_comparison_df['cover_type']].shape[0]/best_comparison_df.shape[0]*100
per_missing_roads=best_comparison_df[best_comparison_df['cover_type']=='undetected'].shape[0]/best_comparison_df.shape[0]*100
per_undeter_roads=best_comparison_df[best_comparison_df['cover_type']=='undetermined'].shape[0]/best_comparison_df.shape[0]*100
per_wrong_roads=round(100-per_right_roads-per_missing_roads-per_undeter_roads,2)

logger.info(f"{round(per_right_roads,2)}% of the roads were found and have the correct road type.")
logger.info(f"{round(per_undeter_roads,2)}% of the roads were detected, but have an undetermined road type.")
logger.info(f"{round(per_missing_roads,2)}% of the roads were not found.")
logger.info(f"{per_wrong_roads}% of the roads had the wrong road type.")

for cover_type in ['undetected', 'undetermined']:
    print('\n')
    per_type_roads_100=round(best_comparison_df[
                                        (best_comparison_df['cover_type']==cover_type) &
                                        (best_comparison_df['CATEGORY']=='artificial')
                                        ].shape[0]/best_comparison_df.shape[0]*100,2)

    per_type_roads_200=round(best_comparison_df[
                                        (best_comparison_df['cover_type']==cover_type) &
                                        (best_comparison_df['CATEGORY']=='natural')
                                        ].shape[0]/best_comparison_df.shape[0]*100,2)

    logger.info(f"{per_type_roads_100}% of the roads are {cover_type} and have the artificial type")
    logger.info(f"{per_type_roads_200}% of the roads are {cover_type} and have the natural type")

print('\n')

# Test for different threshold on the difference between indices
logger.info('Searching for the optimal threshold on the difference between indices...')
filtered_metrics_by_class=pd.DataFrame()
filtered_global_metrics=pd.DataFrame()

tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

for threshold in thresholds:
    tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

    filtered_results=best_comparison_df.copy()
    filtered_results.drop(columns=['tag'], inplace=True)
    filtered_results.loc[filtered_results['diff_score']<threshold, 'cover_type']='undetermined'
    filtered_results['tag']=filtered_results.apply(lambda row: get_tag(row), axis=1)
    part_metrics_by_class, part_global_metrics = get_balanced_accuracy(filtered_results, CLASSES)

    part_metrics_by_class['threshold']=threshold
    part_global_metrics['threshold']=threshold

    filtered_metrics_by_class=pd.concat([filtered_metrics_by_class, part_metrics_by_class], ignore_index=True)
    filtered_global_metrics=pd.concat([filtered_global_metrics, part_global_metrics], ignore_index=True)

    if threshold==0:
        best_filtered_threshold=0
        best_filtered_results=filtered_results
        # max_f1=part_global_metrics.f1w[0]
        max_f1=part_global_metrics.f1b[0]
        
        best_by_class_filtered_metrics=part_metrics_by_class
        best_global_filtered_metrics=part_global_metrics

    # elif (part_global_metrics.f1w>max_f1)[0]:
    elif (part_global_metrics.f1b>max_f1)[0]:
        best_filtered_threshold=threshold
        best_filtered_results=filtered_results
        # max_f1=part_global_metrics.f1w[0]
        max_f1=part_global_metrics.f1b[0]
        
        best_by_class_filtered_metrics=part_metrics_by_class
        best_global_filtered_metrics=part_global_metrics
        print('\n')
        logger.info(f"The best threshold of the difference of indices for the f1-score is now {best_filtered_threshold}.")

    tqdm_log.update(1)

tqdm_log.close()

print('\n')
logger.info(f"For a threshold on the difference of indices of {best_filtered_threshold}...")
show_metrics(best_by_class_filtered_metrics, best_global_filtered_metrics)

filepath=os.path.join(shp_gpkg_folder, 'filtered_types_from_detections.shp')
best_filtered_results.to_file(filepath)
written_files.append(filepath)

print('\n')

# Filters from data exploration
# logger.info('Applying filters from data exploration...')
# comp_df_explo=best_comparison_df.copy()

# comp_df_explo.loc[comp_df_explo['road_len']>1300, 'cover_type']='artificial'

# comp_df_explo.drop(columns=['tag'], inplace=True)
# comp_df_explo['tag']=comp_df_explo.apply(lambda row: get_tag(row), axis=1)

# class_metrics_post_explo, global_metrics_post_explo=get_balanced_accuracy(comp_df_explo, CLASSES)
# show_metrics(class_metrics_post_explo, global_metrics_post_explo)

# print('\n')

# If all roads where classified as artificial (baseline)
if True:
    logger.info('If all roads were classified as artificial...')
    comp_df_all_art=best_comparison_df.copy()
    comp_df_all_art['cover_type']='artificial'
    comp_df_all_art.drop(columns=['tag'], inplace=True)
    comp_df_all_art['tag']=comp_df_all_art.apply(lambda row: get_tag(row), axis=1)

    class_metrics_all_art, global_metrics_all_art=get_balanced_accuracy(comp_df_all_art, CLASSES)
    show_metrics(class_metrics_all_art, global_metrics_all_art)
    print('\n')

# Get the bin accuracy
logger.info('Calculate the bin accuracy to estimate the calibration...')
accuracy_tables=[]
bin_accuracy_param={'artificial':['art_score', 'artificial', 'artifical score'],
                    'natural': ['nat_score', 'natural', 'natural score'], 
                    'artificial_diff': ['diff_score', 'artificial', 'score diff in artificial roads'],
                    'naturall_diff':['diff_score', 'natural', 'score diff in natural roads']}
thresholds_bins=np.arange(0, 1.05, 0.05)
for param in bin_accuracy_param.keys():
    bin_values=[]
    threshold_values=[]
    for threshold in thresholds_bins:
        roads_in_bin=best_filtered_results[
                                        (best_filtered_results[bin_accuracy_param[param][0]]>threshold-0.5) &
                                        (best_filtered_results[bin_accuracy_param[param][0]]<=threshold) &
                                        (best_filtered_results['CATEGORY']==bin_accuracy_param[param][1])
                                        ]

        if not roads_in_bin.empty:
            bin_values.append(
                    roads_in_bin[
                        roads_in_bin['cover_type']==bin_accuracy_param[param][1]
                        ].shape[0]/roads_in_bin.shape[0])
            threshold_values.append(threshold)

    df=pd.DataFrame({'threshold': threshold_values, 'accuracy': bin_values})
    df.name=bin_accuracy_param[param][2]
    accuracy_tables.append(df)

# Make the graphs
# Code strongly inspired from the script 'assess_predictions.py' in the object detector.
logger.info('Make some graphs for the visualization of the impact from the thresholds...')
images_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'images'))

fig = go.Figure()
fig_k = go.Figure()

# Plot of the precision vs recall
fig.add_trace(
    go.Scatter(
        x=all_global_metrics['Rw'],
        y=all_global_metrics['Pw'],
        mode='markers+lines',
        text=all_global_metrics['threshold'],
        name='weighted aggregation'
    )
)

fig.add_trace(
    go.Scatter(
        x=all_global_metrics['Rb'],
        y=all_global_metrics['Pb'],
        mode='markers+lines',
        text=all_global_metrics['threshold'],
        name='balanced aggregation'
    )
)

fig.update_layout(
    xaxis_title="Recall",
    yaxis_title="Precision",
    xaxis=dict(range=[0., 1]),
    yaxis=dict(range=[0., 1])
)

file_to_write = os.path.join(images_folder, 'precision_vs_recall.html')
fig.write_html(file_to_write)
written_files.append(file_to_write)

if len(CLASSES)>1:
    for id_cl in CLASSES:

        fig_k.add_trace(
            go.Scatter(
                x=all_metrics_by_class['Rk'][all_metrics_by_class['cover_class']==id_cl],
                y=all_metrics_by_class['Pk'][all_metrics_by_class['cover_class']==id_cl],
                mode='markers+lines',
                text=all_metrics_by_class['threshold'][all_metrics_by_class['cover_class']==id_cl],
                name=str(id_cl) + ' roads'
            )
        )

    fig_k.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0., 1]),
        yaxis=dict(range=[0., 1])
    )

    file_to_write = os.path.join(images_folder, 'precision_vs_recall_dep_on_class.html')
    fig_k.write_html(file_to_write)
    written_files.append(file_to_write)

# Plot the number of TP, FN, and FPs
fig = go.Figure()

for id_cl in CLASSES:
    
    for y in ['TP', 'FN', 'FP']:

        fig.add_trace(
            go.Scatter(
                x=all_metrics_by_class['threshold'][all_metrics_by_class['cover_class']==id_cl],
                y=all_metrics_by_class[y][all_metrics_by_class['cover_class']==id_cl],
                mode='markers+lines',
                name=y[0:2]+'_'+str(id_cl)
            )
        )

    fig.update_layout(xaxis_title="threshold", yaxis_title="#")
    
if len(CLASSES)>1:
    file_to_write = os.path.join(images_folder, f'TP-FN-FP_vs_threshold_dep_on_class.html')

else:
    file_to_write = os.path.join(images_folder, f'TP-FN-FP_vs_threshold.html')

fig.write_html(file_to_write)
written_files.append(file_to_write)

# Plot the metrics vs thresholds
fig = go.Figure()

for y in ['Pw', 'Rw', 'f1w', 'Pb', 'Rb', 'f1b']:

    fig.add_trace(
        go.Scatter(
            x=all_global_metrics['threshold'],
            y=all_global_metrics[y],
            mode='markers+lines',
            name=y
        )
    )

fig.update_layout(xaxis_title="threshold")

file_to_write = os.path.join(images_folder, f'metrics_vs_threshold.html')
fig.write_html(file_to_write)
written_files.append(file_to_write)

# Plot the number of Pk, Rk dep on class and threshold on the final score
fig = go.Figure()

for id_cl in CLASSES:
    
    for y in ['Pk', 'Rk']:

        fig.add_trace(
            go.Scatter(
                x=filtered_metrics_by_class['threshold'][filtered_metrics_by_class['cover_class']==id_cl],
                y=filtered_metrics_by_class[y][filtered_metrics_by_class['cover_class']==id_cl],
                mode='markers+lines',
                name=y[0:2]+'_'+str(id_cl)
            )
        )

    fig.update_layout(xaxis_title="threshold")
    
file_to_write = os.path.join(images_folder, f'metrics_vs_final_score_threshold_dep_on_class.html')
fig.write_html(file_to_write)
written_files.append(file_to_write)

# Make the calibration curve
fig=go.Figure()

for trace in accuracy_tables:
    fig.add_trace(
        go.Scatter(
            x=trace['threshold'],
            y=trace['accuracy'],
            mode='markers+lines',
            name=trace.name,
        )
    )

fig.add_trace(
    go.Scatter(
        x=thresholds_bins,
        y=thresholds_bins,
        mode='lines',
        name='reference',
    )
)

fig.update_layout(xaxis_title="confidance threshold", yaxis_title="bin accuracy", title="Reliability diagram")

file_to_write = os.path.join(images_folder, f'reliability_diagram.html')
fig.write_html(file_to_write)
written_files.append(file_to_write)

logger.info('The following files were written:')
for file in written_files:
    logger.info(file)