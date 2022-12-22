import os, sys
import yaml
import logging, argparse

import pandas as pd
import geopandas as gpd
import numpy as np

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
OD_FOLDER=os.path.join(PROCESSED_FOLDER, cfg['object_detector_folder'])

GROUND_TRUTH=os.path.join(PROCESSED_FOLDER, cfg['input']['ground_truth'])
PREDICTIONS=cfg['input']['to_evaluate']
CONSIDERED_TILES=os.path.join(OD_FOLDER, cfg['input']['considered_tiles'])

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


    metrics_dict={'class':[], 'TP':[], 'FP':[], 'FN':[], 'Pk':[], 'Rk':[], 'count':[]}
    for road_type in CLASSES:
        metrics_dict['class'].append(road_type)
        tp=comparison_df[(comparison_df['CATEGORY']==comparison_df['road_type']) &
                        (comparison_df['CATEGORY']==road_type)].shape[0]
        fp=comparison_df[(comparison_df['CATEGORY']!=comparison_df['road_type']) &
                        (comparison_df['road_type']==road_type)].shape[0]
        fn=comparison_df[(comparison_df['CATEGORY']!=comparison_df['road_type']) &
                        (comparison_df['CATEGORY']==road_type)].shape[0]

        metrics_dict['TP'].append(tp)
        metrics_dict['FP'].append(fp)
        metrics_dict['FN'].append(fn)

        if tp==0:
            pk=0
            rk=0
        else:
            pk=tp/(tp+fp)
            rk=tp/(tp+fn)

        metrics_dict['Pk'].append(pk)
        metrics_dict['Rk'].append(rk)

        metrics_dict['count'].append(comparison_df[comparison_df['CATEGORY']==road_type].shape[0])

    metrics_df=pd.DataFrame(metrics_dict)

    total_roads_by_type=metrics_df['count'].sum()

    precision=(metrics_df['Pk']*metrics_df['count']).sum()/total_roads_by_type
    recall=(metrics_df['Rk']*metrics_df['count']).sum()/total_roads_by_type

    if precision==0 and recall==0:
        f1_score=0
    else:
        f1_score=round(2*precision*recall/(precision + recall), 2)

    global_metrics_df=pd.DataFrame({'P': precision, 'R': recall, 'F1-score': f1_score})

    return metrics_df, global_metrics_df

def get_corresponding_class(row):
    if row['pred_class']==0:
        return 'artificial'
    elif row['pred_class']==1:
        return 'natural'
    else:
        print(f"Unexpected class: {row['pred_class']}")
        sys.exit(1)

# Importing files ----------------------------------
print('Importing files...')

ground_truth=gpd.read_file(GROUND_TRUTH)

predictions=gpd.GeoDataFrame()
for dataset_name in PREDICTIONS.values():
    dataset=gpd.read_file(os.path.join(OD_FOLDER, dataset_name))
    predictions=pd.concat([predictions, dataset], ignore_index=True)
predictions['pred_class_name']=predictions.apply(lambda row: get_corresponding_class(row), axis=1)

considered_tiles=gpd.read_file(CONSIDERED_TILES)

quarries=gpd.read_file(os.path.join(INITIAL_FOLDER, 'created/quarries.shp'))

# Information treatment ----------------------------
print('Limiting the labels to the visible area...')

tiles_union=considered_tiles['geometry'].unary_union
considered_zone=gpd.GeoDataFrame({'id_tiles_union': [i for i in range(len(tiles_union.geoms))],
                                'geometry': [geo for geo in tiles_union.geoms]},
                                crs=4326
                                )

fct_misc.test_crs(considered_zone.crs, ground_truth)
visible_ground_truth=gpd.overlay(ground_truth, considered_zone, how="intersection")

print('Getting the intersecting area...')

ground_truth_2056=visible_ground_truth.to_crs(epsg=2056)
ground_truth_2056['area_label']=ground_truth_2056.area

predictions_2056=predictions.to_crs(epsg=2056)

fct_misc.test_crs(ground_truth_2056.crs, predictions_2056.crs)
predicted_roads_2056=gpd.overlay(ground_truth_2056, predictions_2056, how='intersection')

predicted_roads_filtered=predicted_roads_2056[(~predicted_roads_2056['OBJECTID'].isna()) &
                                            (~predicted_roads_2056['score'].isna())].copy()
predicted_roads_filtered['joined_area']=predicted_roads_filtered.area
predicted_roads_filtered['area_pred_in_label']=round(predicted_roads_filtered['joined_area']/predicted_roads_filtered['area_label'], 2)
predicted_roads_filtered['weighted_score']=predicted_roads_filtered['area_pred_in_label']*predicted_roads_filtered['score']


print('Calculating the indexes and the metrics per threshold for the score of the predictions...')

all_global_metrics=pd.DataFrame()
all_metrics_by_class=pd.DataFrame()

thresholds=np.arange(0, 1., 0.05)
tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

for threshold in thresholds:
    tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

    final_type={'road_id':[], 'road_type':[], 'nat_score':[], 'art_score':[], 'diff_score':[]}
    valid_pred_roads=predicted_roads_filtered[predicted_roads_filtered['score']>=threshold]
    detected_roads_id=valid_pred_roads['OBJECTID'].unique().tolist()

    for road_id in ground_truth['OBJECTID'].unique().tolist():

        if road_id not in detected_roads_id:
            final_type['road_id'].append(road_id)
            final_type['road_type'].append('undetected')
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
            final_type['road_type'].append('undetermined')
            final_type['diff_score'].append(0)
        elif artificial_index > natural_index:
            final_type['road_id'].append(road_id)
            final_type['road_type'].append('artificial')
            final_type['diff_score'].append(abs(artificial_index-natural_index))
        elif artificial_index < natural_index:
            final_type['road_id'].append(road_id)
            final_type['road_type'].append('natural')
            final_type['diff_score'].append(abs(artificial_index-natural_index))

        final_type['art_score'].append(round(artificial_index,3))
        final_type['nat_score'].append(round(natural_index, 3))

    final_type_df=pd.DataFrame(final_type)

    comparison_df=gpd.GeoDataFrame(final_type_df.merge(ground_truth[['OBJECTID','geometry', 'CATEGORY', 'road_len']],
                                    how='inner', left_on='road_id', right_on='OBJECTID'))
    try:
        assert(comparison_df.shape[0]==ground_truth.shape[0]), "There are too many or not enough labels in the final results"
    except Exception as e:
        print(e)
        sys.exit(1)

    tags=[]
    for road in comparison_df.itertuples():
        pred_class=road.road_type
        gt_class=road.CATEGORY

        if pred_class=='undetermined' or pred_class=='undetected':
            tags.append('FN')
        elif pred_class==gt_class:
            tags.append('TP')
        else:
            tags.append('FP')
    comparison_df['tag']=tags

    part_metrics_by_class, part_global_metrics = get_balanced_accuracy(comparison_df, CLASSES)

    part_metrics_by_class['threshold']=threshold
    part_global_metrics['threshold']=threshold

    all_metrics_by_class=pd.concat([all_metrics_by_class, part_metrics_by_class], ignore_index=True)
    all_global_metrics=pd.concat([all_global_metrics, part_global_metrics], ignore_index=True)

    if threshold==0:
        all_preds_comparison_df=comparison_df
        max_f1=part_global_metrics.f1_score
        best_threshold=0

        for metric in part_metrics_by_class.itertuples():
            print(f"The {metric.road_type} roads have a precision of {round(metric.pk, 2)}",
                f" and a recall of {round(metric.rk, 2)}")

        print(f"The final F1-score for a threshold of 0 is {part_global_metrics.f1_score}", 
            f" with a precision of {round(part_global_metrics.precision,2)} and a recall of",
            f" {round(part_global_metrics.recall,2)}.")

    elif part_global_metrics.f1_score>max_f1:
        best_threshold=threshold
        best_comparison_df=comparison_df
        max_f1=part_global_metrics.f1_score
        
        best_by_class_metrics=part_metrics_by_class
        best_global_metrics=part_global_metrics

        print(f"The best threshold for the f1-score is now {best_threshold}.")

    tqdm_log.update(1)

tqdm_log.close()

# print(f"The best threshold for the f1-score is at {best_threshold}.")
print(f"For a threshold of {best_threshold}...")

for metric in best_by_class_metrics.itertuples():
    print(f"The {metric.road_type} roads have a precision of {round(metric.pk, 2)}",
        f" and a recall of {round(metric.rk, 2)}")

print(f"The final F1-score is {best_global_metrics.f1_score}", 
    f" with a precision of {round(best_global_metrics.precision,2)}", 
    f" and a recall of {round(best_global_metrics.recall,2)}.")

shp_gpkg_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'shp_gpkg'))
filename='types_from_detections.shp'
best_comparison_df.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append('final/shp_gpkg/' + filename)

filename='types_from_all_detections.shp'
all_preds_comparison_df.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append('final/shp_gpkg/' + filename)


print('-- Calculating the accuracy...')

per_right_roads=best_comparison_df[best_comparison_df['CATEGORY']==best_comparison_df['road_type']].shape[0]/best_comparison_df.shape[0]*100
per_missing_roads=best_comparison_df[best_comparison_df['road_type']=='undetected'].shape[0]/best_comparison_df.shape[0]*100
per_undeter_roads=best_comparison_df[best_comparison_df['road_type']=='undetermined'].shape[0]/best_comparison_df.shape[0]*100
per_wrong_roads=round(100-per_right_roads-per_missing_roads-per_undeter_roads,2)

print(f"{round(per_right_roads,2)}% of the roads were found and have the correct road type.")
print(f"{round(per_undeter_roads,2)} of the roads were detected, but have an undetermined road type.")
print(f"{round(per_missing_roads,2)}% of the roads were not found.")
print(f"{per_wrong_roads}% of the roads had the wrong road type.")

for road_type in ['undetected', 'undetermined']:
    print('\n')
    per_type_roads_100=round(best_comparison_df[
                                        (best_comparison_df['road_type']==road_type) &
                                        (best_comparison_df['CATEGORY']=='artificial')
                                        ].shape[0]/best_comparison_df.shape[0]*100,2)

    per_type_roads_200=round(best_comparison_df[
                                        (best_comparison_df['road_type']==road_type) &
                                        (best_comparison_df['CATEGORY']=='natural')
                                        ].shape[0]/best_comparison_df.shape[0]*100,2)

    print(f"{per_type_roads_100}% of the roads are {road_type} and have the artificial type")
    print(f"{per_type_roads_200}% of the roads are {road_type} and have the natural type")

print('\n')

# Test for different threshold on the difference between indices
print('Searching for the optimal threshold on the difference between indices...')
filtered_metrics_by_class=pd.DataFrame()
filtered_global_metrics=pd.DataFrame()

tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

for threshold in thresholds:
    tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

    filtered_results=best_comparison_df[best_comparison_df['diff_score']>=threshold].copy()
    part_metrics_by_class, part_global_metrics = get_balanced_accuracy(filtered_results, CLASSES)

    part_metrics_by_class['threshold']=threshold
    part_global_metrics['threshold']=threshold

    filtered_metrics_by_class=pd.concat([filtered_metrics_by_class, part_metrics_by_class], ignore_index=True)
    filtered_global_metrics=pd.concat([filtered_global_metrics, part_global_metrics], ignore_index=True)

    if threshold==0:
        max_f1=part_global_metrics.f1_score
        best_filtered_threshold=0

    elif part_global_metrics.f1_score>max_f1:
        best_filtered_threshold=threshold
        best_filtered_results=filtered_results
        max_f1=part_global_metrics.f1_score
        
        best_by_class_filtered_metrics=part_metrics_by_class
        best_global_filtered_metrics=part_global_metrics

        print(f"The best threshold of the difference of indices for the f1-score is now {best_filtered_threshold}.")

    tqdm_log.update(1)

tqdm_log.close()

# print(f"The best threshold for the f1-score is at {best_threshold}.")
print(f"For a threshold of the difference of indices of {best_filtered_threshold}...")

for metric in best_by_class_filtered_metrics.itertuples():
    print(f"The {metric.road_type} roads have a precision of {round(metric.pk, 2)}",
        f" and a recall of {round(metric.rk, 2)}")

print(f"The final F1-score is {best_global_filtered_metrics.f1_score}", 
    f" with a precision of {round(best_global_filtered_metrics.precision,2)}", 
    f" and a recall of {round(best_global_filtered_metrics.recall,2)}.")

shp_gpkg_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'shp_gpkg'))
filename='filtered_types_from_detections.shp'
best_comparison_df.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append('final/shp_gpkg/' + filename)

filename='filtered_types_from_all_detections.shp'
all_preds_comparison_df.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append('final/shp_gpkg/' + filename)

print('/n')

# Roads in quarries are always naturals
print('Checking for roads in quarries...')

quarries_4326=quarries.to_crs(epsg=4326)

fct_misc.test_crs(best_comparison_df.crs, quarries_4326.crs)

roads_in_quarries=gpd.sjoin(best_comparison_df, quarries_4326, predicate='within')

comp_df_quarries=best_comparison_df.copy()
for road_id in roads_in_quarries['OBJECTID'].unique().tolist():
    comp_df_quarries.loc[comp_df_quarries['OBJECTID']==road_id, 'road_type']='natural'

class_metrics_post_quarries, global_metrics_post_quarries=get_balanced_accuracy(comp_df_quarries, CLASSES)

print(f"For a threshold of {best_threshold}...")

for metric in class_metrics_post_quarries.itertuples():
    print(f"The {metric.road_type} roads have a precision of {round(metric.pk, 2)}",
        f" and a recall of {round(metric.rk, 2)}")

print(f"The final F1-score is {global_metrics_post_quarries.f1_score}", 
    f" with a precision of {round(global_metrics_post_quarries.precision,2)}", 
    f" and a recall of {round(global_metrics_post_quarries.recall,2)}.")

print('\n')

# Filters from data exploration
print('Applying filters from data exploration...')
comp_df_explo=comp_df_quarries.copy()

comp_df_explo.loc[comp_df_quarries['diff_score']<=0.06, 'road_type']='undetermined'


comp_df_explo.loc[comp_df_quarries['road_len']>1300, 'road_type']='artificial'

class_metrics_post_explo, global_metrics_post_explo=get_balanced_accuracy(comp_df_explo, CLASSES)

for metric in class_metrics_post_explo.itertuples():
    print(f"The {metric.road_type} roads have a precision of {round(metric.pk, 2)}",
        f" and a recall of {round(metric.rk, 2)}")

print(f"The final F1-score is {global_metrics_post_explo.f1_score}", 
    f" with a precision of {round(global_metrics_post_explo.precision,2)}", 
    f" and a recall of {round(global_metrics_post_explo.recall,2)}.")

print('\n')

# If all roads where classified as artificial (baseline)
if False:
    print('\n')
    print('If all roads were classified as artificial...')
    comp_df_all_art=best_comparison_df.copy()
    comp_df_all_art['road_type']='artificial'

    class_metrics_all_art, global_metrics_all_art=get_balanced_accuracy(comp_df_all_art, CLASSES)

    for metric in class_metrics_all_art.itertuples():
        print(f"The {metric.road_type} roads have a precision of {round(metric.pk, 2)}",
            f" and a recall of {round(metric.rk, 2)}")

    print(f"The final F1-score is {global_metrics_all_art.f1_score}", 
        f" with a precision of {round(global_metrics_all_art.precision,2)}", 
        f" and a recall of {round(global_metrics_all_art.recall,2)}.")

print('The following files were written:')
for file in written_files:
    print(file)