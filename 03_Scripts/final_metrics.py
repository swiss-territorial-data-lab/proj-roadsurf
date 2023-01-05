import os, sys
import yaml
import logging, argparse

import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from tqdm import tqdm

import fct_misc


with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['final_metrics.py']    #  [os.path.basename(__file__)]


# Define constants ------------------------------------

DEBUG_MODE=cfg['debug_mode']
CLASSES=['artificial', 'natural']

INITIAL_FOLDER=cfg['initial_folder']
PROCESSED_FOLDER=cfg['processed_folder']
FINAL_FOLDER=cfg['final_folder']
# OD_FOLDER=os.path.join(PROCESSED_FOLDER, cfg['object_detector_folder'])

ROAD_PARAMETERS=os.path.join(INITIAL_FOLDER, cfg['input']['road_param'])
GROUND_TRUTH=os.path.join(PROCESSED_FOLDER, cfg['input']['ground_truth'])
if 'layer' in cfg['input'].keys():
    LAYER=cfg['input']['layer']
PREDICTIONS=cfg['input']['to_evaluate']
# CONSIDERED_TILES=os.path.join(OD_FOLDER, cfg['input']['considered_tiles'])
CONSIDERED_TILES=os.path.join(FINAL_FOLDER, cfg['input']['considered_tiles'])

shp_gpkg_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'shp_gpkg'))

written_files=[]

# Definition of functions ---------------------------

def get_balanced_accuracy(comparison_df, CLASSES):
    '''
    Get a dataframe with the GT and the predictions and calculate the per-class and blanced-weighted precision, accuracy
    and f1-score.

    - comparison_df: dataframe with the GT and the predictions
    - CLASSES: classes to search for

    return: a dataframe with the TP, FP, FN, precision and recall per class and a second one with the global metrics.
    '''

    metrics_dict={'cover_class':[], 'TP':[], 'FP':[], 'FN':[], 'Pk':[], 'Rk':[], 'count':[]}
    for cover_type in CLASSES:
        metrics_dict['cover_class'].append(cover_type)
        tp=comparison_df[(comparison_df['tag']=='TP') &
                        (comparison_df['CATEGORY']==cover_type)].shape[0]
        fp_precision=comparison_df[(comparison_df['tag']=='FP') &
                        (comparison_df['cover_type']==cover_type)].shape[0]
        fp_recall=comparison_df[(comparison_df['tag']=='FP') &
                        (comparison_df['CATEGORY']==cover_type)].shape[0]
        fn=comparison_df[(comparison_df['tag']=='FN') &
                        (comparison_df['CATEGORY']==cover_type)].shape[0]

        metrics_dict['TP'].append(tp)
        metrics_dict['FP'].append(fp_precision)
        metrics_dict['FN'].append(fn+fp_recall)

        if tp==0:
            pk=0
            rk=0
        else:
            pk=tp/(tp+fp_precision)
            rk=tp/(tp+fn+fp_recall)

        metrics_dict['Pk'].append(pk)
        metrics_dict['Rk'].append(rk)

        metrics_dict['count'].append(comparison_df[comparison_df['CATEGORY']==cover_type].shape[0])

    metrics_df=pd.DataFrame(metrics_dict)

    total_roads_by_type=metrics_df['count'].sum()

    precision=(metrics_df['Pk']*metrics_df['count']).sum()/total_roads_by_type
    recall=(metrics_df['Rk']*metrics_df['count']).sum()/total_roads_by_type

    if precision==0 and recall==0:
        f1_score=0
    else:
        f1_score=round(2*precision*recall/(precision + recall), 2)
    global_metrics_df=pd.DataFrame({'precision': [precision], 'recall': [recall], 'f1_score': [f1_score]})

    return metrics_df, global_metrics_df

def get_corresponding_class(row):
    'Get the class in words from the class ids out of the object detector with the method apply.'

    if row['pred_class']==0:
        return 'artificial'
    elif row['pred_class']==1:
        return 'natural'
    else:
        print(f"Unexpected class: {row['pred_class']}")
        sys.exit(1)

def determine_category(row):
    'Get the class in words from the codes out of the swissTLM3D with the method apply.'

    if row['BELAGSART']==100:
        return 'artificial'
    if row['BELAGSART']==200:
        return 'natural'
    else:
        print(f"Unexpected class: {row['BELAGSART']}")
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
             return 'FP'
        else:
            print(f'Unexpected configuration: prediction class is {pred_class} and ground truth class is {gt_class}.')
            sys.exit(1)

def show_metrics(metrics_by_class, global_metrics):
    '''
    Print the by class precision and recall and the global precision, recall and f1-score

    - metrics_by_class: The by-class metrics as given by the function get_balanced_accuracy()
    - global_metrics: The global metrics as given by the function get_balanced_accuracy()

    return: -
    '''

    for metric in metrics_by_class.itertuples():
        print(f"The {metric.cover_class} roads have a precision of {round(metric.Pk, 2)}",
            f"and a recall of {round(metric.Rk, 2)}")

    print(f"The final f1-score is {global_metrics.f1_score[0]}", 
        f"with a precision of {round(global_metrics.precision[0],2)} and a recall of",
        f"{round(global_metrics.recall[0],2)}.")


# Importing files ----------------------------------
print('Importing files...')

road_parameters=pd.read_excel(ROAD_PARAMETERS)

# ground_truth=gpd.read_file(GROUND_TRUTH, layer=LAYER)
ground_truth=gpd.read_file(GROUND_TRUTH)

predictions=gpd.GeoDataFrame()
for dataset_name in PREDICTIONS.values():
    # dataset=gpd.read_file(os.path.join(OD_FOLDER, dataset_name))
    dataset=gpd.read_file(os.path.join(FINAL_FOLDER, dataset_name))
    predictions=pd.concat([predictions, dataset], ignore_index=True)
predictions['pred_class_name']=predictions.apply(lambda row: get_corresponding_class(row), axis=1)
predictions.drop(columns=['pred_class'], inplace=True)
del dataset

considered_tiles=gpd.read_file(CONSIDERED_TILES)

quarries=gpd.read_file(os.path.join(INITIAL_FOLDER, 'created/quarries.shp'))

# Information treatment ----------------------------
print('Filtering the GT for the roads of interest...')
filtered_road_parameters=road_parameters[road_parameters['to keep']=='yes'].copy()
filtered_ground_truth=ground_truth.merge(filtered_road_parameters[['GDB-Code','Width']], 
                                        how='inner',left_on='OBJEKTART',right_on='GDB-Code')
filtered_ground_truth=filtered_ground_truth[filtered_ground_truth['BELAGSART']!=999997]

filtered_ground_truth['CATEGORY']=filtered_ground_truth.apply(lambda row: determine_category(row), axis=1)

# Roads in quarries are always naturals
print('-- Roads in quarries are always naturals...')

buffered_quarries=quarries.copy()
buffered_quarries['geometry']=buffered_quarries.buffer(5)
buffered_quarries_4326=buffered_quarries.to_crs(epsg=4326)

fct_misc.test_crs(filtered_ground_truth.crs, buffered_quarries_4326.crs)

roads_in_quarries=gpd.sjoin(filtered_ground_truth, buffered_quarries_4326, predicate='within')
filename='roads_in_quarries.shp'
roads_in_quarries.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append(os.path.join(shp_gpkg_folder, filename))

filtered_ground_truth=filtered_ground_truth[~filtered_ground_truth['OBJECTID'].isin(
                                    roads_in_quarries['OBJECTID'].unique().tolist())] 

print('Limiting the labels to the visible area of labels and predictions...')

tiles_union=considered_tiles['geometry'].unary_union
considered_zone=gpd.GeoDataFrame({'id_tiles_union': [i for i in range(len(tiles_union.geoms))],
                                'geometry': [geo for geo in tiles_union.geoms]},
                                crs=4326
                                )

fct_misc.test_crs(considered_zone.crs, filtered_ground_truth)
visible_ground_truth=gpd.overlay(filtered_ground_truth, considered_zone, how="intersection")

del considered_tiles, tiles_union, considered_zone, 

print('Getting the intersecting area between predictions and labels...')

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

print('Calculating the indexes and the metrics per threshold for the score of the predictions...')

all_global_metrics=pd.DataFrame()
all_metrics_by_class=pd.DataFrame()

thresholds=np.arange(0, 1., 0.05)
tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

for threshold in thresholds:
    tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

    final_type={'road_id':[], 'cover_type':[], 'nat_score':[], 'art_score':[], 'diff_score':[]}
    valid_pred_roads=predicted_roads_filtered[predicted_roads_filtered['score']>=threshold]
    detected_roads_id=valid_pred_roads['OBJECTID'].unique().tolist()

    for road_id in visible_ground_truth['OBJECTID'].unique().tolist():

        if road_id not in detected_roads_id:
            final_type['road_id'].append(road_id)
            final_type['cover_type'].append('undetected')
            final_type['nat_score'].append(0)
            final_type['art_score'].append(0)
            final_type['diff_score'].append(0)
            continue

        intersecting_predictions=valid_pred_roads[valid_pred_roads['OBJECTID']==road_id].copy()

        groups=intersecting_predictions.groupby(['pred_class_name']).sum()
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

    comparison_df=gpd.GeoDataFrame(final_type_df.merge(visible_ground_truth[['OBJECTID','geometry', 'CATEGORY']],
                                    how='inner', left_on='road_id', right_on='OBJECTID'))
    try:
        assert(comparison_df.shape[0]==visible_ground_truth.shape[0]), "There are too many or not enough labels in the final results"
    except Exception as e:
        print(e)
        sys.exit(1)

    comparison_df['tag']=comparison_df.apply(lambda row: get_tag(row), axis=1)

    part_metrics_by_class, part_global_metrics = get_balanced_accuracy(comparison_df, CLASSES)

    part_metrics_by_class['threshold']=threshold
    part_global_metrics['threshold']=threshold

    all_metrics_by_class=pd.concat([all_metrics_by_class, part_metrics_by_class], ignore_index=True)
    all_global_metrics=pd.concat([all_global_metrics, part_global_metrics], ignore_index=True)

    balanced_P=part_metrics_by_class['Pk'].sum()/2
    balanced_R=part_metrics_by_class['Rk'].sum()/2
    if balanced_P==0 and balanced_R==0:
        balanced_f1=0
    else:
        balanced_f1=2*balanced_P*balanced_R/(balanced_P + balanced_R)

    if threshold==0:
        all_preds_comparison_df=comparison_df

        best_threshold=0
        best_comparison_df=comparison_df
        # max_f1=part_global_metrics.f1_score[0]
        max_f1=balanced_f1
        
        best_by_class_metrics=part_metrics_by_class
        best_global_metrics=part_global_metrics

        print('\n')
        show_metrics(part_metrics_by_class, part_global_metrics)

    # elif (part_global_metrics.f1_score>max_f1)[0]:
    elif (balanced_f1>max_f1):
        best_threshold=threshold
        best_comparison_df=comparison_df
        # max_f1=part_global_metrics.f1_score[0]
        max_f1=balanced_f1
        
        best_by_class_metrics=part_metrics_by_class
        best_global_metrics=part_global_metrics

        print('\n')
        print(f"The best threshold for the f1-score is now {best_threshold}.")

    # else:
    #     print(part_global_metrics.f1_score[0])

    tqdm_log.update(1)

tqdm_log.close()

print('\n')
# print(f"The best threshold for the f1-score is at {best_threshold}.")
print(f"For a threshold of {best_threshold}...")
show_metrics(best_by_class_metrics, best_global_metrics)

filename='types_from_detections.shp'
best_comparison_df.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append('final/shp_gpkg/' + filename)

filename='types_from_all_detections.shp'
all_preds_comparison_df.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append('final/shp_gpkg/' + filename)

print('\n')
print('-- Calculating the accuracy...')

per_right_roads=best_comparison_df[best_comparison_df['CATEGORY']==best_comparison_df['cover_type']].shape[0]/best_comparison_df.shape[0]*100
per_missing_roads=best_comparison_df[best_comparison_df['cover_type']=='undetected'].shape[0]/best_comparison_df.shape[0]*100
per_undeter_roads=best_comparison_df[best_comparison_df['cover_type']=='undetermined'].shape[0]/best_comparison_df.shape[0]*100
per_wrong_roads=round(100-per_right_roads-per_missing_roads-per_undeter_roads,2)

print(f"{round(per_right_roads,2)}% of the roads were found and have the correct road type.")
print(f"{round(per_undeter_roads,2)} of the roads were detected, but have an undetermined road type.")
print(f"{round(per_missing_roads,2)}% of the roads were not found.")
print(f"{per_wrong_roads}% of the roads had the wrong road type.")

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

    print(f"{per_type_roads_100}% of the roads are {cover_type} and have the artificial type")
    print(f"{per_type_roads_200}% of the roads are {cover_type} and have the natural type")

print('\n')

# Test for different threshold on the difference between indices
print('Searching for the optimal threshold on the difference between indices...')
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

    balanced_P=part_metrics_by_class['Pk'].sum()/2
    balanced_R=part_metrics_by_class['Rk'].sum()/2
    if balanced_P==0 and balanced_R==0:
        balanced_f1=0
    else:
        balanced_f1=2*balanced_P*balanced_R/(balanced_P + balanced_R)

    if threshold==0:
        best_filtered_threshold=0
        best_filtered_results=filtered_results
        # max_f1=part_global_metrics.f1_score[0]
        max_f1=balanced_f1
        
        best_by_class_filtered_metrics=part_metrics_by_class
        best_global_filtered_metrics=part_global_metrics

    # elif (part_global_metrics.f1_score>max_f1)[0]:
    elif (balanced_f1>max_f1):
        best_filtered_threshold=threshold
        best_filtered_results=filtered_results
        # max_f1=part_global_metrics.f1_score[0]
        max_f1=balanced_f1
        
        best_by_class_filtered_metrics=part_metrics_by_class
        best_global_filtered_metrics=part_global_metrics
        print('\n')
        print(f"The best threshold of the difference of indices for the f1-score is now {best_filtered_threshold}.")

    tqdm_log.update(1)

tqdm_log.close()

print('\n')
# print(f"The best threshold for the f1-score is at {best_threshold}.")
print(f"For a threshold of the difference of indices of {best_filtered_threshold}...")
show_metrics(best_by_class_filtered_metrics, best_global_filtered_metrics)

shp_gpkg_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'shp_gpkg'))
filename='filtered_types_from_detections.shp'
best_filtered_results.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append('final/shp_gpkg/' + filename)

print('\n')

# Filters from data exploration
print('Applying filters from data exploration...')
comp_df_explo=best_comparison_df.copy()

# comp_df_explo.loc[comp_df_explo['road_len']>1300, 'cover_type']='artificial'

# comp_df_explo.drop(columns=['tag'], inplace=True)
# comp_df_explo['tag']=comp_df_explo.apply(lambda row: get_tag(row), axis=1)

class_metrics_post_explo, global_metrics_post_explo=get_balanced_accuracy(comp_df_explo, CLASSES)
show_metrics(class_metrics_post_explo, global_metrics_post_explo)

print('\n')

# If all roads where classified as artificial (baseline)
if True:
    print('If all roads were classified as artificial...')
    comp_df_all_art=best_comparison_df.copy()
    comp_df_all_art['cover_type']='artificial'
    comp_df_all_art.drop(columns=['tag'], inplace=True)
    comp_df_all_art['tag']=comp_df_all_art.apply(lambda row: get_tag(row), axis=1)

    class_metrics_all_art, global_metrics_all_art=get_balanced_accuracy(comp_df_all_art, CLASSES)
    show_metrics(class_metrics_all_art, global_metrics_all_art)
    print('\n')

# Get the bin accuracy
print('Calculate the bin accuracy to estimate the calibration...')
accuracy_tables=[]
bin_accuracy_param={'artificial':['art_score', 'artificial', 'artifical score'],
                    'natural': ['nat_score', 'natural', 'natural score'], 
                    'artificial_diff': ['diff_score', 'artificial', 'score diff in artificial roads'],
                    'naturall_diff':['diff_score', 'natural', 'score diff in natural roads']}
for param in bin_accuracy_param.keys():
    bin_values=[]
    threshold_values=[]
    for threshold in thresholds[1:]:
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
print('Make some graphs for the visualization of the impact from the thresholds...')
images_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'images'))

fig = go.Figure()
fig_k = go.Figure()

# Plot of the precision vs recall
fig.add_trace(
    go.Scatter(
        x=all_global_metrics['recall'],
        y=all_global_metrics['precision'],
        mode='markers+lines',
        text=all_global_metrics['threshold'],
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

for y in ['precision', 'recall', 'f1_score']:

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

# Make the calibratin curve
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
        x=thresholds,
        y=thresholds,
        mode='lines',
        name='reference',
    )
)

fig.update_layout(xaxis_title="confidance threshold", yaxis_title="bin accuracy", title="Reliability diagram")

file_to_write = os.path.join(images_folder, f'reliability_diagram.html')
fig.write_html(file_to_write)
written_files.append(file_to_write)

# Make histograms of the score depending on the tag.
parameters={'artificial': 'art_score',
            'natural': 'nat_score'}

for cover_type in parameters.keys():
    results=best_filtered_results[best_filtered_results['cover_type']==cover_type]
    hist_artificial=results.plot.hist(column=[parameters[cover_type]], by='tag',
                                    range=(0, 1.0),
                                    bins=20,
                                    title=f'Repartition of the class score depending on the tag',
                                    figsize=(8,6),
                                    grid=True, 
                                    ec='black',
                                    )
    fig = hist_artificial[0].get_figure()
    fig.tight_layout() 
    file_to_write = os.path.join(images_folder, f'histogram_{cover_type}_scores_per_tag.jpeg')
    fig.savefig(file_to_write, bbox_inches='tight')
    written_files.append(file_to_write)

determined_roads=best_filtered_results[~best_filtered_results['cover_type'].isin(['undetermined', 'undetected'])]
hist_diff_score=determined_roads.plot.hist(column=['diff_score'], by=['tag', 'cover_type'],
                                range=(0, 1.0),
                                bins=20,
                                title=f'Repartition of the class score depending on the tag',
                                figsize=(5, 10),
                                grid=True,
                                ec='black',
                                )
fig = hist_diff_score[0].get_figure()
file_to_write = os.path.join(images_folder, f'histogram_diff_scores_per_tag_and_cover.jpeg')
fig.tight_layout() 
fig.savefig(file_to_write, bbox_inches='tight')
written_files.append(file_to_write)


print('The following files were written:')
for file in written_files:
    print(file)