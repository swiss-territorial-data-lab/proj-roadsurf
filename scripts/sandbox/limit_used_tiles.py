import os
import sys
import logging, logging.config
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
from functions.fct_misc import ensure_dir_exists

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

tic = time()
logger.info('Starting...')

config_file='config/config_od.yaml'
logger.info(f"Using {config_file} as config file.")

with open(config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


CONSIDERED_DATASETS = cfg['considered_datasets']

WORKING_DIR=cfg['working_directory']
INPUT_DIR=cfg['input_directory']
OUPUT_DIR=cfg['output_directory']

GROUND_TRUTH = cfg['ground_truth_labels']
OTHER_LABELS = cfg['other_labels']
TILES = cfg['tiles']

os.chdir(WORKING_DIR)
_ = ensure_dir_exists(OUPUT_DIR)


logger.info('Read the dataset...')
ground_truth_gdf = gpd.read_file(GROUND_TRUTH)
other_labels_gdf = gpd.read_file(OTHER_LABELS)
tiles_gdf = gpd.read_file(TILES)

logger.info('Select the tiles')
possible_sets_for_training = ['trn', 'tst', 'val', 'gt', 'ground truth']
if any([key_word in CONSIDERED_DATASETS for key_word in possible_sets_for_training]) and ('oth' not in CONSIDERED_DATASETS):
    considered_labels_gdf = ground_truth_gdf.copy()
    extra_labels_gdf = other_labels_gdf.copy()
elif all([key_word not in CONSIDERED_DATASETS for key_word in possible_sets_for_training]) and ('oth' in CONSIDERED_DATASETS):
    considered_labels_gdf = other_labels_gdf.copy()
    extra_labels_gdf = ground_truth_gdf.copy()
else:
    logger.error('Unclear indications regarding the datasets to consider.')
    sys.exit(1)

potential_tiles_gdf = gpd.sjoin(tiles_gdf, considered_labels_gdf, how = 'left')
potential_tiles_gdf.drop_duplicates(subset = ['id'], inplace = True)
excluded_tiles_gdf = gpd.sjoin(tiles_gdf, extra_labels_gdf, how = 'inner')
excluded_id_list = excluded_tiles_gdf.id.unique()

considered_tiles_gdf = potential_tiles_gdf.loc[~potential_tiles_gdf.id.isin(excluded_id_list), tiles_gdf.columns].reset_index(drop=True)

filepath='tiles_aoi.geojson'
considered_tiles_gdf.to_file(filepath)
logger.info(f'Done! The file {filepath} was written.')

toc = time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")