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

PROCESSED_FOLDER=cfg['processed_folder']
FINAL_FOLDER=cfg['final_folder']
OD_FOLDER=os.path.join(PROCESSED_FOLDER, cfg['object_detector_folder'])

GROUND_TRUTH=cfg['input']['ground_truth']
PREDICTIONS=cfg['input']['to_evaluate']

# Importing files ----------------------------------
print('Importing files...')

ground_truth=gpd.read_file(os.path.join(PROCESSED_FOLDER, GROUND_TRUTH))

predictions=gpd.GeoDataFrame()
for dataset_name in PREDICTIONS.values():
    dataset=gpd.read_file(os.path.join(OD_FOLDER, dataset_name))
    predictions=pd.concat([predictions, dataset], ignore_index=True)

# Information treatment ----------------------------

# Only roads smaller than 4 m are naturals


# Roads in quarries are always naturals


