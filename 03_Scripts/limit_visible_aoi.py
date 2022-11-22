import os, sys
import argparse, yaml
import logging

from tqdm import tqdm
from glob import glob

import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

import rasterio
from rasterio.mask import mask

import numpy as np

import fct_misc

# Get the configuration
# parser = argparse.ArgumentParser(description="This script prepares datasets for the determination of the road cover type.")
# parser.add_argument('config_file', type=str, help='a YAML config file')
# args = parser.parse_args()

with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['limit_visible_aoi.py']    #  [os.path.basename(__file__)]

# Define constants ------------------------------------------------------
DATA_FOLDER = cfg['data_folder']

IMAGES_FOLDER_IN = os.path.join(DATA_FOLDER, cfg['input']['images'])
IMAGES_FOLDER_OUT = os.path.join(DATA_FOLDER, cfg['output']['images'])

ROADS=os.path.join(DATA_FOLDER, cfg['input']['roads'])
ROADS_PARAMETER=os.path.join(DATA_FOLDER, cfg['input']['roads_param'])
BELAGSART_TO_KEEP=[100, 200]

GROUND_TRUTH=os.path.join(DATA_FOLDER, cfg['input']['ground_truth'])

ZOOM=cfg['zoom']

fct_misc.ensure_dir_exists(IMAGES_FOLDER_OUT)

# Import files ----------------------------------------------------------
print('Importing files...')

all_files=glob(IMAGES_FOLDER_IN+'/*.tif')

roads=gpd.read_file(ROADS)
roads_parameters=pd.read_excel(ROADS_PARAMETER)

ground_truth=gpd.read_file(GROUND_TRUTH)

# Data treatment ----------------------------------------------------------
print('Determining the restricted area of interest...')

roads_parameters_filtered=roads_parameters[roads_parameters['to keep']=='yes'].copy()
roads_parameters_filtered.drop_duplicates(subset='GDB-Code',inplace=True)       # Keep first by default 

roads_of_interest=roads.merge(roads_parameters_filtered[['GDB-Code']], how='right',left_on='OBJEKTART',right_on='GDB-Code')
roads_of_interest=roads_of_interest[roads_of_interest['BELAGSART'].isin(BELAGSART_TO_KEEP)]

roi_3857=roads_of_interest.to_crs(epsg=3857)

buffered_roads=roi_3857.unary_union.buffer(20)
restricted_AOI = gpd.GeoDataFrame({'id_labels':[x for x in range(len(buffered_roads.geoms))],
                                    'geometry':[geo for geo in buffered_roads.geoms]},
                                    crs="EPSG:3857")

# restricted_AOI.to_file(os.path.join(DATA_FOLDER, 'processed/shapefiles_gpkg/test_restricted_AOI.shp'))

# extract the geometry in GeoJSON format
geoms = [mapping(buffered_roads)]

print('Filtering the tiles of interest...')
files=[fp for fp in all_files if '/'+ str(ZOOM)+'_' in fp]

print('Limiting the images to the restricted aoi...')
print(f'The new images are saved in {IMAGES_FOLDER_OUT}.')
for tile_filepath in tqdm(files[100:150], desc='Image treatment'):

    # extract the raster values values within the polygon 
    with rasterio.open(tile_filepath) as src:
        out_image, _ = mask(src, geoms, crop=False)
        profile=src.profile

        # fct_misc.test_crs(roi_3857.crs, src.crs)

    filename=tile_filepath.split('/')[-1]
    dst_filepath=os.path.join(IMAGES_FOLDER_OUT, filename)

    with rasterio.open(dst_filepath, 'w', **profile) as dst:
        dst.write(out_image)

print('Making the labels correspond:')
restricted_AOI_4326 = restricted_AOI.to_crs(epsg=4326)

fct_misc.test_crs(restricted_AOI_4326.crs, ground_truth.crs)


new_labels_name='processed/json/restricted_groundtruth.json'
restricted_labels=ground_truth.overlay(restricted_AOI_4326)
restricted_labels.to_file(os.path.join(DATA_FOLDER, new_labels_name))

print(f'New labels written in {new_labels_name}')