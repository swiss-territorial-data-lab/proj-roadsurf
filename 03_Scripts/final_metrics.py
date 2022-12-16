import os, sys
import yaml
import logging, argparse

import pandas as pd
import geopandas as gpd

import fct_misc


with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['final_metrics.py']    #  [os.path.basename(__file__)]


# Define constants ------------------------------------

DEBUG_MODE=cfg['debug_mode']
THRESHOLD=cfg['threshold']
CLASSES=['artificial', 'natural']

INITIAL_FOLDER=cfg['initial_folder']
PROCESSED_FOLDER=cfg['processed_folder']
FINAL_FOLDER=cfg['final_folder']
OD_FOLDER=os.path.join(PROCESSED_FOLDER, cfg['object_detector_folder'])

GROUND_TRUTH=os.path.join(PROCESSED_FOLDER, cfg['input']['ground_truth'])
PREDICTIONS=cfg['input']['to_evaluate']

written_files=[]

# Definition of functions ---------------------------

def get_balanced_accuracy(comparison_df, CLASSES):
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

        print(f"The {road_type} roads have a precision of {round(pk, 2)} and a recall of {round(rk, 2)}")

    metrics_df=pd.DataFrame(metrics_dict)

    total_roads_by_type=metrics_df['count'].sum()

    precision=(metrics_df['Pk']*metrics_df['count']).sum()/total_roads_by_type
    recall=(metrics_df['Rk']*metrics_df['count']).sum()/total_roads_by_type

    if precision==0 and recall==0:
        f1_score=0
    else:
        f1_score=round(2*precision*recall/(precision + recall), 2)

    print(f"The final F1-score for a threshold of {THRESHOLD} is {f1_score}", 
        f" with a precision of {round(precision,2)} and a recall of {round(recall,2)}.")

    return metrics_df

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

quarries=gpd.read_file(os.path.join(INITIAL_FOLDER, 'created/quarries.shp'))

# Information treatment ----------------------------
print('Getting the intersecting area...')

ground_truth_2056=ground_truth.to_crs(epsg=2056)
ground_truth_2056['area_label']=ground_truth_2056.area

predictions_2056=predictions.to_crs(epsg=2056)

fct_misc.test_crs(ground_truth_2056.crs, predictions_2056.crs)
predicted_roads_2056=gpd.overlay(ground_truth_2056, predictions_2056, how='intersection')

predicted_roads_filtered=predicted_roads_2056[(~predicted_roads_2056['OBJECTID'].isna()) &
                                            (~predicted_roads_2056['score'].isna())].copy()
predicted_roads_filtered['joined_area']=predicted_roads_filtered.area
predicted_roads_filtered['area_pred_in_label']=round(predicted_roads_filtered['joined_area']/predicted_roads_filtered['area_label'], 2)
predicted_roads_filtered['weighted_score']=predicted_roads_filtered['area_pred_in_label']*predicted_roads_filtered['score']


print('Caclulating the indexes...')

final_type={'road_id':[], 'road_type':[], 'nat_score':[], 'art_score':[]}
detected_roads_id=predicted_roads_filtered['OBJECTID'].unique().tolist()

for road_id in ground_truth['OBJECTID'].unique().tolist():

    if road_id not in detected_roads_id:
        final_type['road_id'].append(road_id)
        final_type['road_type'].append('undetermined')
        final_type['nat_score'].append(0)
        final_type['art_score'].append(0)
        continue

    intersecting_predictions=predicted_roads_filtered[predicted_roads_filtered['OBJECTID']==road_id].copy()

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
    elif artificial_index > natural_index:
        final_type['road_id'].append(road_id)
        final_type['road_type'].append('artificial')
    elif artificial_index < natural_index:
        final_type['road_id'].append(road_id)
        final_type['road_type'].append('natural')
    else:
        final_type['road_id'].append(road_id)
        final_type['road_type'].append('undetermined')

    final_type['art_score'].append(round(artificial_index,3))
    final_type['nat_score'].append(round(natural_index, 3))

final_type_df=pd.DataFrame(final_type)

comparison_df=gpd.GeoDataFrame(final_type_df.merge(ground_truth[['OBJECTID','geometry', 'CATEGORY']], how='inner',
                            left_on='road_id', right_on='OBJECTID'))
try:
    comparison_df.shape[0]==ground_truth.shape[0], "There are to many or not enough labels in the final results"
except Exception as e:
    print(e)
    sys.exit(1)

tags=[]
for road in comparison_df.itertuples():
    pred_class=road.road_type
    gt_class=road.CATEGORY

    if pred_class=='undetermined':
        tags.append('FN')
    elif pred_class==gt_class:
        tags.append('TP')
    else:
        tags.append('FP')
comparison_df['tag']=tags

shp_gpkg_folder=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'shp_gpkg'))
filename='types_from_detections.shp'
comparison_df.to_file(os.path.join(shp_gpkg_folder, filename))
written_files.append('final/shp_gpkg/' + filename)

print('Calculating the metrics...')

print('-- Calculating the accuracy...')

per_right_roads=round(comparison_df[comparison_df['CATEGORY']==comparison_df['road_type']].shape[0]/comparison_df.shape[0]*100,2)
per_missing_roads=round(comparison_df[comparison_df['road_type']=='undetermined'].shape[0]/comparison_df.shape[0]*100,2)
per_wrong_roads=round(100-per_right_roads-per_missing_roads,2)

print(f"{per_right_roads}% of the roads were found and have the correct road type.")

print(f"{per_missing_roads}% of the roads were not found.")
print(f"{per_wrong_roads}% of the roads had the wrong road type.")

per_missing_roads_100=round(comparison_df[
                                        (comparison_df['road_type']=='undetermined') &
                                        (comparison_df['CATEGORY']=='artificial')
                                        ].shape[0]/comparison_df.shape[0]*100,2)

per_missing_roads_200=round(comparison_df[
                                        (comparison_df['road_type']=='undetermined') &
                                        (comparison_df['CATEGORY']=='natural')
                                        ].shape[0]/comparison_df.shape[0]*100,2)

print(f"{per_missing_roads_100}% of the roads are missing and have the artificial type")
print(f"{per_missing_roads_200}% of the roads are missing and have the natural type")


print('-- Calculating the macro balanced weighted accuracy...')
metrics=get_balanced_accuracy(comparison_df, CLASSES)


# Only roads smaller than 4 m are naturals


# Roads in quarries are always naturals
print('Checking for roads in quarries...')

quarries_4326=quarries.to_crs(epsg=4326)

fct_misc.test_crs(comparison_df.crs, quarries_4326.crs)

roads_in_quarries=gpd.sjoin(comparison_df, quarries_4326, predicate='within')

comp_df_quarries=comparison_df.copy()
for road_id in roads_in_quarries['OBJECTID'].unique().tolist():
    comp_df_quarries.loc[comp_df_quarries['OBJECTID']==road_id, 'road_type']='natural'

metrics_post_quarries=get_balanced_accuracy(comp_df_quarries, CLASSES)

# All the undetermined are artificial
print('Considering all the undetermined roads as artifical roads...')
comp_df_max_art=comp_df_quarries.copy()
comp_df_max_art['road_type']=comp_df_max_art['road_type'].replace('undetermined', 'artificial')

metrics_mar_art=get_balanced_accuracy(comp_df_max_art, CLASSES)

# If all roads where classified as artificial
if False:
    print('If all roads were classified as artificial...')
    comp_df_all_art=comparison_df.copy()
    comp_df_all_art['road_type']='artificial'

    metrics_all_art=get_balanced_accuracy(comp_df_all_art, CLASSES)

print('The following files were written:')
for file in written_files:
    print(file)