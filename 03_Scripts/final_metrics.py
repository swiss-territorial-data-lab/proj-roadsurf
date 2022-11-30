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

PROCESSED_FOLDER=cfg['processed_folder']
FINAL_FOLDER=cfg['final_folder']
OD_FOLDER_100=os.path.join(PROCESSED_FOLDER, cfg['object_detector_folder_100'])
OD_FOLDER_200=os.path.join(PROCESSED_FOLDER, cfg['object_detector_folder_200'])

GROUND_TRUTH_100=cfg['input']['ground_truth_100']
GROUND_TRUTH_200=cfg['input']['ground_truth_200']
PREDICTIONS=cfg['input']['to_evaluate']

# Importing files ----------------------------------
print('Importing files...')

ground_truth_100=gpd.read_file(os.path.join(PROCESSED_FOLDER, GROUND_TRUTH_100))
ground_truth_200=gpd.read_file(os.path.join(PROCESSED_FOLDER, GROUND_TRUTH_200))

predictions_100=gpd.GeoDataFrame()
for dataset_name in PREDICTIONS.values():
    dataset=gpd.read_file(os.path.join(OD_FOLDER_100, dataset_name))
    predictions_100=pd.concat([predictions_100, dataset], ignore_index=True)

predictions_200=gpd.GeoDataFrame()
for dataset_name in PREDICTIONS.values():
    dataset=gpd.read_file(os.path.join(OD_FOLDER_200, dataset_name))
    predictions_200=pd.concat([predictions_200, dataset], ignore_index=True)

# Information treatment ----------------------------
print('Formatting the data...')

fct_misc.test_crs(ground_truth_100.crs, ground_truth_200.crs)
ground_truth_100['CATEGORY']="artificial"
ground_truth_200['CATEGORY']="natural"
ground_truth=pd.concat([ground_truth_100, ground_truth_200], ignore_index=True)
ground_truth['SUPERCATEGORY']="road"

predictions_100['CATEGORY']="artificial"
predictions_200['CATEGORY']="natural"
predictions=pd.concat([predictions_100, predictions_200], ignore_index=True)

print('Getting the intersecting area...')

ground_truth_2056=ground_truth.to_crs(epsg=2056)
ground_truth_2056['area_label']=ground_truth_2056.area

predictions_2056=predictions.to_crs(epsg=2056)
predictions_2056['area_predictions']=predictions_2056.area

fct_misc.test_crs(ground_truth_2056.crs, predictions_2056.crs)
predicted_roads_2056=gpd.overlay(ground_truth_2056, predictions_2056, how='intersection')

predicted_roads_filtered=predicted_roads_2056[(~predicted_roads_2056['OBJECTID'].isna()) &
                                            (~predicted_roads_2056['score'].isna())].copy()
predicted_roads_filtered['joined_area']=predicted_roads_filtered.area
predicted_roads_filtered['area_pred_in_label']=round(predicted_roads_filtered['joined_area']/predicted_roads_filtered['area_label']*100, 2)
predicted_roads_filtered['weighted_score']=predicted_roads_filtered['area_pred_in_label']*predicted_roads_filtered['score']

print('Caclulating the indexes...')

final_type={'road_id':[], 'road_type':[]}
detected_roads_id=predicted_roads_filtered['OBJECTID'].unique().tolist()

for road_id in ground_truth['OBJECTID'].unique().tolist():

    if road_id not in detected_roads_id:
        final_type['road_id'].append(road_id)
        final_type['road_type'].append('undetermined')
        continue

    intersecting_predictions=predicted_roads_filtered[predicted_roads_filtered['OBJECTID']==road_id].copy()

    groups=intersecting_predictions.groupby(['CATEGORY_2']).sum()
    if 'natural' in groups.index:
        natural_index=groups.loc['natural', 'weighted_score']/groups.loc['natural', 'score']
    else:
        natural_index=0
    if 'artificial' in groups.index:
        artificial_index=groups.loc['artificial', 'weighted_score']/groups.loc['artificial', 'score']
    else:
        artificial_index=0

    if artificial_index > natural_index:
        final_type['road_id'].append(road_id)
        final_type['road_type'].append('artificial')
    elif artificial_index < natural_index:
        final_type['road_id'].append(road_id)
        final_type['road_type'].append('natural')
    else:
        final_type['road_id'].append(road_id)
        final_type['road_type'].append('undetermined')

final_type_df=pd.DataFrame(final_type)

comparison_df=final_type_df.merge(ground_truth[['OBJECTID', 'CATEGORY']], how='inner',
                                left_on='road_id', right_on='OBJECTID')
try:
    comparison_df.shape[0]==ground_truth.shape[0], "There are to many or not enough labels in the final results"
except Exception as e:
    print(e)
    sys.exit(1)


print('Calculating the metrics...')

print('-- Calculating the accuracy...')

per_right_roads=round(comparison_df[comparison_df['CATEGORY']==comparison_df['road_type']].shape[0]/comparison_df.shape[0]*100,2)
per_missing_roads=round(comparison_df[comparison_df['road_type']=='undetermined'].shape[0]/comparison_df.shape[0]*100,2)
per_wrong_roads=round(100-per_right_roads-per_missing_roads,2)

print(f"{per_right_roads}% of the roads were found and have the correct road type.")

print(f"{per_missing_roads}% of the roads were not found.")
print(f"{per_wrong_roads}% of the roads had the wrong road type.")


print('-- Calculating the macro balanced weighted accuracy...')

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

    pk=tp/(tp+fp)
    rk=tp/(tp+fn)
    metrics_dict['Pk'].append(pk)
    metrics_dict['Rk'].append(rk)

    metrics_dict['count'].append(comparison_df[comparison_df['CATEGORY']==road_type].shape[0])

    print(f"The {road_type} roads have a precision of {round(pk, 2)} and a recall of {round(rk, 2)}")

metrics_df=pd.DataFrame(metrics_dict)

total_pixels=metrics_df['count'].sum()

precision=(metrics_df['Pk']*metrics_df['count']).sum()/total_pixels
recall=(metrics_df['Rk']*metrics_df['count']).sum()/total_pixels

f1_score=round(2*precision*recall/(precision + recall), 2)

print(f"The final F1-score for a threshold of {THRESHOLD} is {f1_score}", 
    f" with a precision of {round(precision,2)} and a recall of {round(recall,2)}.")

# Only roads smaller than 4 m are naturals


# Roads in quarries are always naturals


