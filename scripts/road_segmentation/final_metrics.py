import os, sys
import yaml
import logging, logging.config
import time

import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from tqdm import tqdm

import determine_class
sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

tic = time.time()
logger.info('Starting...')

logger.info(f"Using config.yaml as config file.")
with open('config/config_od.yaml') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['final_metrics.py']


# Define constants ------------------------------------

INITIAL_FOLDER = cfg['initial_folder']
PROCESSED_FOLDER = cfg['processed_folder']
FINAL_FOLDER=cfg['final_folder']

BASELINE = cfg['baseline']

ROAD_PARAMETERS = os.path.join(INITIAL_FOLDER, cfg['inputs']['road_param'])

GROUND_TRUTH = os.path.join(PROCESSED_FOLDER, cfg['inputs']['ground_truth'])
if 'other_labels' in cfg['inputs'].keys():
    OTHER_LABELS = os.path.join(PROCESSED_FOLDER, cfg['inputs']['other_labels'])
else:
    OTHER_LABELS = None

PREDICTIONS = cfg['inputs']['to_evaluate']
TILES = os.path.join(PROCESSED_FOLDER, cfg['inputs']['tiles'])
LABELS_ID = os.path.join(PROCESSED_FOLDER, cfg['inputs']['labels_id'])

QUARRIES = os.path.join(INITIAL_FOLDER, cfg['inputs']['quarries'])


shp_gpkg_folder = fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'shp_gpkg'))

written_files=[]

# Definition of functions ---------------------------

def get_metrics(comparison_df, CLASSES):
    '''
    Get a dataframe with the GT, the predictions and the tags (TP, FP, FN)
    Calculate the per-class and global precision, recall and f1-score.

    - comparison_df: dataframe with the GT and the predictions
    - CLASSES: classes to search for

    return: a dataframe with the TP, FP, FN, precision and recall per class and a second one with the global metrics.
    '''

    metrics_dict={'cover_class':[], 'TP':[], 'FP':[], 'FN':[], 'Pk':[], 'Rk':[], 'f1k': [], 'count':[]}
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
            pk = 0
            rk = 0
            f1k = 0
        else:
            pk = tp/(tp+fp)
            rk = tp/(tp+fn+fn_class)
            f1k = 2*pk*rk/(pk+rk)

        metrics_dict['Pk'].append(pk)
        metrics_dict['Rk'].append(rk)
        metrics_dict['f1k'].append(f1k)

        metrics_dict['count'].append(comparison_df[comparison_df['CATEGORY']==cover_type].shape[0])

    metrics_df=pd.DataFrame(metrics_dict)

    total_by_type=metrics_df['count'].sum()

    weighted_precision=(metrics_df['Pk']*metrics_df['count']).sum()/total_by_type
    weighted_recall=(metrics_df['Rk']*metrics_df['count']).sum()/total_by_type

    if weighted_precision==0 and weighted_recall==0:
        weighted_f1_score=0
    else:
        weighted_f1_score=2*weighted_precision*weighted_recall/(weighted_precision + weighted_recall)

    balanced_precision=metrics_df['Pk'].sum()/2
    balanced_recall=metrics_df['Rk'].sum()/2

    if balanced_precision==0 and balanced_recall==0:
        balanced_f1_score=0
    else:
        balanced_f1_score=2*balanced_precision*balanced_recall/(balanced_precision + balanced_recall)

    global_metrics_df=pd.DataFrame({'Pw': [weighted_precision], 'Rw': [weighted_recall], 'f1w': [weighted_f1_score],
                                    'Pb': [balanced_precision], 'Rb': [balanced_recall], 'f1b': [balanced_f1_score]})

    return metrics_df, global_metrics_df

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

    - metrics_by_class: The by-class metrics as given by the function get_metrics()
    - global_metrics: The global metrics as given by the function get_metrics()

    return: -
    '''

    for metric in metrics_by_class.itertuples():
        logger.info('%s %s',
            f"The {metric.cover_class} roads have a precision of {round(metric.Pk, 2)}",
            f"and a recall of {round(metric.Rk, 2)}")

    logger.info('%s %s %s',
        f"The final f1-score is {round(global_metrics.f1b[0], 2)}", 
        f"with a precision of {round(global_metrics.Pb[0],2)} and a recall of",
        f"{round(global_metrics.Rb[0],2)}.")


# Importing files ----------------------------------
logger.info('Importing files...')

road_parameters=pd.read_excel(ROAD_PARAMETERS)

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
predictions['pred_class_name']=predictions.apply(lambda row: determine_class.get_corresponding_class(row, labels_id),
                                                    axis=1)
predictions.drop(columns=['pred_class'], inplace=True)

tiles=gpd.read_file(TILES)
considered_tiles=tiles[tiles['dataset'].isin(PREDICTIONS.keys())]

quarries=gpd.read_file(QUARRIES)

del dataset, tiles

def from_preds_to_metrics(predictions, ground_truth, by_class_metrics, global_metrics, dataset_name, threshold=0, show=False):

    comparison_df=determine_class.determine_detected_class(predictions, ground_truth, threshold)

    comparison_df['tag']=comparison_df.apply(lambda row: get_tag(row), axis=1)

    dst_metrics_by_class, dst_global_metrics = get_metrics(comparison_df, CLASSES)

    if show:
        show_metrics(dst_metrics_by_class, dst_global_metrics)

    dst_metrics_by_class['dataset'] = dataset_name
    dst_metrics_by_class['threshold'] = threshold
    by_class_metrics = pd.concat([by_class_metrics, dst_metrics_by_class], ignore_index=True)

    dst_global_metrics['dataset'] = dataset_name
    dst_global_metrics['threshold'] = threshold
    global_metrics = pd.concat([global_metrics, dst_global_metrics], ignore_index=True)

    return comparison_df, by_class_metrics, global_metrics


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

filtered_ground_truth['CATEGORY']=filtered_ground_truth.apply(lambda row: determine_class.determine_category(row), axis=1)

# Roads in quarries are always naturals
logger.info('-- Roads in quarries are always naturals...')

roads_in_quarries, filtered_ground_truth = determine_class.get_roads_in_quarries(quarries, filtered_ground_truth)
filepath=os.path.join(shp_gpkg_folder, 'roads_in_quarries.shp')
roads_in_quarries.to_file(filepath)
written_files.append(os.path.join(filepath))

logger.info('Limiting the labels to the visible area of labels and predictions...')

visible_ground_truth=determine_class.clip_labels(filtered_ground_truth, considered_tiles[['title', 'id', 'geometry']])

logger.info('Getting the intersecting area between predictions and labels...')

visible_ground_truth_2056=visible_ground_truth.to_crs(epsg=2056)
predictions_2056=predictions.to_crs(epsg=2056)

predicted_roads_filtered=determine_class.get_weighted_scores(visible_ground_truth_2056, predictions_2056)
predicted_roads_filtered.drop(columns=['OBJEKTART', 'KUNSTBAUTE', 'BELAGSART', 'road_width', 'road_len',
                                       'CATEGORY', 'SUPERCATEGORY', 'gt_type', 'GDB-Code', 'Width',
                                        'title', 'tile_id', 'area_label', 'crs', 'dataset', 'joined_area'])

del visible_ground_truth_2056, ground_truth, predictions_2056

logger.info('Determining the best metrics for the predictions based on the validation dataset...')
val_predictions=predicted_roads_filtered[predicted_roads_filtered['dataset']=='val']
validation_tiles=considered_tiles[considered_tiles['dataset']=='val']
validation_ground_truth=filtered_ground_truth[filtered_ground_truth.geometry.intersects(validation_tiles.unary_union)]

all_global_metrics=pd.DataFrame()
all_metrics_by_class=pd.DataFrame()

thresholds=np.arange(0, 1., 0.05)
tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

for threshold in thresholds:
    tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

    val_comparison_df=determine_class.determine_detected_class(val_predictions, validation_ground_truth, threshold)

    val_comparison_df['tag']=val_comparison_df.apply(lambda row: get_tag(row), axis=1)

    part_metrics_by_class, part_global_metrics = get_metrics(val_comparison_df, CLASSES)

    part_metrics_by_class['threshold']=threshold
    part_global_metrics['threshold']=threshold

    all_metrics_by_class=pd.concat([all_metrics_by_class, part_metrics_by_class], ignore_index=True)
    all_global_metrics=pd.concat([all_global_metrics, part_global_metrics], ignore_index=True)

    if threshold==0:
        best_threshold=0
        max_f1=part_global_metrics.f1b[0]
        max_P=part_global_metrics.Pb[0]

        best_val_by_class_metrics=part_metrics_by_class
        best_val_global_metrics=part_global_metrics

    elif (part_global_metrics.f1b>max_f1)[0] or ((part_global_metrics.f1b==max_f1)[0] and (part_global_metrics.Pb>max_P)[0]):
        best_threshold=round(threshold,2)
        max_f1=part_global_metrics.f1b[0]
        max_P=part_global_metrics.Pb[0]
        
        best_val_by_class_metrics=part_metrics_by_class
        best_val_global_metrics=part_global_metrics

        print('\n')
        logger.info(f"The best threshold for the f1-score is now {best_threshold}.")

    tqdm_log.update(1)

tqdm_log.close()

print('\n')
logger.info("Metrics for the validation dataset:")
show_metrics(best_val_by_class_metrics, best_val_global_metrics)

by_class_metrics=best_val_by_class_metrics.copy()
by_class_metrics['dataset']='val'
global_metrics=best_val_global_metrics.copy()
global_metrics['dataset']='val'

print('\n')
logger.info(f"For a threshold of {best_threshold}...")

comparison_df, by_class_metrics, global_metrics = from_preds_to_metrics(predicted_roads_filtered, filtered_ground_truth, 
                                                                    by_class_metrics, global_metrics, 
                                                                    'all datasets', best_threshold, show = True)

try:
    assert(comparison_df.shape[0]==filtered_ground_truth.shape[0]), "There are too many or not enough labels in the final results"
except Exception as e:
    logger.error(e)
    sys.exit(1)

best_comparison_df=comparison_df.copy()

filepath=os.path.join(shp_gpkg_folder, 'types_from_detections.shp')
best_comparison_df.to_file(filepath)
written_files.append(filepath)

print('\n')
logger.info('Metrics based on the trn, tst, val datasets...')

for dst in ['trn', 'tst']:
    dst_predictions = predicted_roads_filtered[predicted_roads_filtered['dataset']==dst]
    dst_tiles = considered_tiles[considered_tiles['dataset']==dst]
    dst_ground_truth = filtered_ground_truth[filtered_ground_truth.geometry.intersects(dst_tiles.unary_union)]

    dst_comparison_df, by_class_metrics, global_metrics = from_preds_to_metrics(dst_predictions, dst_ground_truth,
                                                                            by_class_metrics, global_metrics,
                                                                            dst, best_threshold)


not_oth_predictions=predicted_roads_filtered[predicted_roads_filtered['dataset'].isin(['trn', 'tst', 'val'])]
ground_truth_from_gt=filtered_ground_truth[filtered_ground_truth['gt_type']=='gt']

not_oth_comparison_df, by_class_metrics, global_metrics = from_preds_to_metrics(not_oth_predictions, ground_truth_from_gt,
                                                                            by_class_metrics, global_metrics,
                                                                            'training zone (trn, val, tst)', best_threshold, show=True)

if 'oth' in PREDICTIONS.keys():
    print('\n')
    logger.info('Metrics based on the predictions of the oth dataset...')

    oth_predictions=predicted_roads_filtered[predicted_roads_filtered['dataset']=='oth']
    ground_truth_from_oth=filtered_ground_truth[filtered_ground_truth['gt_type']=='oth']

    oth_comparison_df, by_class_metrics, global_metrics = from_preds_to_metrics(oth_predictions, ground_truth_from_oth,
                                                                            by_class_metrics, global_metrics,
                                                                            'inference-only zone', best_threshold, show=True)

if best_threshold != 0:
    print('\n')
    logger.info(f"If we were to keep all the predictions, the metrics would be...")
    all_preds_comparison_df, by_class_metrics, global_metrics = from_preds_to_metrics(predicted_roads_filtered, filtered_ground_truth,
                                                                                  by_class_metrics, global_metrics,
                                                                                  'all predictions without filter', show = True)

    filepath=os.path.join(shp_gpkg_folder, 'types_from_all_detections.shp')
    all_preds_comparison_df.to_file(filepath)
    written_files.append(filepath)


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


logger.info('Searching for the optimal threshold on the difference between indices...')
gt_filtered_metrics_by_class=pd.DataFrame()
gt_filtered_global_metrics=pd.DataFrame()
oth_filtered_metrics_by_class=pd.DataFrame()
oth_filtered_global_metrics=pd.DataFrame()

tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

for threshold in thresholds:
    tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

    filtered_results=best_comparison_df.copy()
    filtered_results.drop(columns=['tag'], inplace=True)
    filtered_results.loc[filtered_results['diff_score']<threshold, 'cover_type']='undetermined'
    filtered_results['tag']=filtered_results.apply(lambda row: get_tag(row), axis=1)

    gt_filtered_results=filtered_results[filtered_results['gt_type']=='gt'].copy()
    gt_part_metrics_by_class, gt_part_global_metrics = get_metrics(gt_filtered_results, CLASSES)

    gt_part_metrics_by_class['threshold']=threshold
    gt_part_global_metrics['threshold']=threshold

    gt_filtered_metrics_by_class=pd.concat([gt_filtered_metrics_by_class, gt_part_metrics_by_class], ignore_index=True)
    gt_filtered_global_metrics=pd.concat([gt_filtered_global_metrics, gt_part_global_metrics], ignore_index=True)

    if  'oth' in PREDICTIONS.keys():
        oth_filtered_results=filtered_results[filtered_results['gt_type']=='oth'].copy()
        oth_part_metrics_by_class, oth_part_global_metrics = get_metrics(oth_filtered_results, CLASSES)

        oth_part_metrics_by_class['threshold']=threshold
        oth_part_global_metrics['threshold']=threshold

        oth_filtered_metrics_by_class=pd.concat([oth_filtered_metrics_by_class, oth_part_metrics_by_class]
                                                , ignore_index=True)
        oth_filtered_global_metrics=pd.concat([oth_filtered_global_metrics, oth_part_global_metrics], 
                                              ignore_index=True)

    if threshold==0:
        best_filtered_threshold=0
        best_filtered_results=filtered_results
        max_f1=gt_part_global_metrics.f1b[0]
        
        best_by_class_filtered_metrics=gt_part_metrics_by_class
        best_global_filtered_metrics=gt_part_global_metrics

    elif (gt_part_global_metrics.f1b>max_f1)[0]:
        best_filtered_threshold=round(threshold,2)
        best_filtered_results=filtered_results
        max_f1=gt_part_global_metrics.f1b[0]
        
        best_by_class_filtered_metrics=gt_part_metrics_by_class
        best_global_filtered_metrics=gt_part_global_metrics
        print('\n')
        logger.info(f"The best threshold of the difference of indices for the f1-score is now {best_filtered_threshold}.")

    tqdm_log.update(1)

tqdm_log.close()
print('\n')

if best_filtered_threshold>0:
    logger.info(f"For a threshold on the difference of indices of {best_filtered_threshold}...")
    show_metrics(best_by_class_filtered_metrics, best_global_filtered_metrics)

    logger.info('%s %s', f'It would be wise to verify all the results with a difference on the indices',
                f'lower than {best_filtered_threshold}.')
    
    filepath=os.path.join(shp_gpkg_folder, 'filtered_types_from_detections.shp')
    best_filtered_results.to_file(filepath)
    written_files.append(filepath)

else:
    logger.info('No threshold on the difference of indices would improve the results.')


print('\n')

# Get baseline
if 'artificial' in BASELINE:
    logger.info('Baseline: If all roads were classified as artificial...')
    comp_df_baseline=best_comparison_df.copy()
    comp_df_baseline['cover_type']='artificial'

elif BASELINE == 'random':
	np.random.seed(0)
	
	logger.info('Baseline: if the roads were classified randomly...')
	comp_df_baseline = best_comparison_df.copy()
	comp_df_baseline['cover_type'] = ['artificial' if i == 1 else 'natural' for i in np.random.randint(1, 3, size=comp_df_baseline.shape[0])]

else:
    logger.critical('No corresponding baseline.')
    sys.exit(1)

comp_df_baseline.drop(columns=['tag'], inplace=True)
comp_df_baseline['tag']=comp_df_baseline.apply(lambda row: get_tag(row), axis=1)

class_metrics_all_art, global_metrics_all_art=get_metrics(comp_df_baseline, CLASSES)
show_metrics(class_metrics_all_art, global_metrics_all_art)

class_metrics_all_art['dataset']='baseline'
by_class_metrics=pd.concat([by_class_metrics, class_metrics_all_art], ignore_index=True)

global_metrics_all_art['dataset']='baseline'
global_metrics=pd.concat([global_metrics, global_metrics_all_art], ignore_index=True)

print('\n')

# Save the metrics for each dataset in csv file.
table_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'tables'))

tmp=by_class_metrics.select_dtypes(include=[np.number])
by_class_metrics.loc[:, tmp.columns] = np.round(tmp, 3)
by_class_metrics.to_csv(os.path.join(table_folder, 'by_class_metrics.csv'))

tmp=global_metrics.select_dtypes(include=[np.number])
global_metrics.loc[:, tmp.columns] = np.round(tmp, 3)
global_metrics.to_csv(os.path.join(table_folder, 'global metrics.csv'))


logger.info('Calculate the bin accuracy to estimate the calibration...')
accuracy_tables=[]
bin_accuracy_param={'artificial':['art_score', 'artificial', 'artifical score'],
                    'natural': ['nat_score', 'natural', 'natural score'], 
                    'artificial_diff': ['diff_score', 'artificial', 'score diff in artificial roads'],
                    'naturall_diff':['diff_score', 'natural', 'score diff in natural roads']}
thresholds_bins=np.arange(0, 1.05, 0.05)

for gt_type in best_comparison_df['gt_type'].unique():
    determined_types=best_comparison_df[best_comparison_df['gt_type']==gt_type].copy()

    for param in bin_accuracy_param.keys():
        bin_values=[]
        threshold_values=[]
        for threshold in thresholds_bins:
            roads_in_bin=determined_types[
                                        (determined_types[bin_accuracy_param[param][0]]>threshold-0.5) &
                                        (determined_types[bin_accuracy_param[param][0]]<=threshold) &
                                        (determined_types['CATEGORY']==bin_accuracy_param[param][1])
                                        ]

            if not roads_in_bin.empty:
                bin_values.append(
                        roads_in_bin[
                            roads_in_bin['cover_type']==bin_accuracy_param[param][1]
                            ].shape[0]/roads_in_bin.shape[0])
                threshold_values.append(threshold)

        df=pd.DataFrame({'threshold': threshold_values, 'accuracy': bin_values})
        df.name=bin_accuracy_param[param][2] + ' for ' + gt_type
        accuracy_tables.append(df)


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

file_to_write = os.path.join(images_folder, 'precision_vs_recall_over_validation_set.html')
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

    file_to_write = os.path.join(images_folder, 'precision_vs_recall_dep_on_class_over_val_set.html')
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
    file_to_write = os.path.join(images_folder, f'TP-FN-FP_vs_threshold_dep_on_class_over_val_set.html')

else:
    file_to_write = os.path.join(images_folder, f'TP-FN-FP_vs_threshold_over_validation_set.html')

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

file_to_write = os.path.join(images_folder, f'metrics_vs_threshold_over_validation_set.html')
fig.write_html(file_to_write)
written_files.append(file_to_write)

# Plot the number of Pk, Rk dep on class and threshold on the final score
fig = go.Figure()

for id_cl in CLASSES:
    
    for y in ['Pk', 'Rk']:

        fig.add_trace(
            go.Scatter(
                x=gt_filtered_metrics_by_class['threshold'][gt_filtered_metrics_by_class['cover_class']==id_cl],
                y=gt_filtered_metrics_by_class[y][gt_filtered_metrics_by_class['cover_class']==id_cl],
                mode='markers+lines',
                name=y[0:2]+'_'+str(id_cl) + '- gt'
            )
        )

        if  'oth' in PREDICTIONS.keys():
            fig.add_trace(
                go.Scatter(
                    x=oth_filtered_metrics_by_class['threshold'][oth_filtered_metrics_by_class['cover_class']==id_cl],
                    y=oth_filtered_metrics_by_class[y][oth_filtered_metrics_by_class['cover_class']==id_cl],
                    mode='markers+lines',
                    name=y[0:2]+'_'+str(id_cl) + '- oth'
                )
            )

    fig.update_layout(xaxis_title="threshold")
    
file_to_write = os.path.join(images_folder, f'metrics_vs_score_diff_threshold_dep_on_class.html')
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