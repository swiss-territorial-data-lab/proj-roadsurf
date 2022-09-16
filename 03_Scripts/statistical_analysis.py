import argparse
import yaml
import os, sys
import time
import logging, logging.config

import pandas as pd
import geopandas as gpd
import rasterio

import stat
import math

from tqdm import tqdm

from misc_fct import test_crs

with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['statistical_analysis.py']    #  [os.path.basename(__file__)]


# Defitions of the functions


# Definition of the constants
INPUT_FOLDER=cfg['input']['input_folder']
PROCESSED=cfg['processed']
PROCESSED_FOLDER=PROCESSED['processed_folder']
FINAL=cfg['final']
FINAL_FOLDER=FINAL['final_folder']

# Inputs
ROADS=PROCESSED_FOLDER + PROCESSED['input_files']['roads']
TILES_DIR=PROCESSED_FOLDER + PROCESSED['input_files']['images']
TILES_INFO=PROCESSED_FOLDER + PROCESSED['input_files']['tiles']

# Outputs
STATS_ROADS=PROCESSED_FOLDER + PROCESSED['output_files']['stats']
STATS_TYPE=FINAL_FOLDER+ FINAL['stats_by_type']


# Importation of the files
roads=gpd.read_file(ROADS)
tiles_info = gpd.read_file(TILES_INFO)


