import os, sys, argparse
import logging, logging.config
import yaml
import time
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

import rasterio
from rasterio.features import rasterize

import numpy as np
import matplotlib.pyplot as plt

import fct_misc

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

tic = time.time()
logger.info('Starting...')

parser = argparse.ArgumentParser(description="This script trains a predictive models.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

# with open('03_Scripts/config.yaml') as fp:
#     cfg = yaml.load(fp, Loader=yaml.FullLoader)['final_metrics.py']

# Define constants ------------------------------
MASK_AS_BAND=cfg['mask_as_band']
if MASK_AS_BAND:
    logger.info('The masks will be written as an additional band on each image.')
else:
    MASK_DIR=cfg['mask_directory']
    logger.info(f'The mask will be written in a separate folder: {MASK_DIR}')
    fct_misc.ensure_dir_exists(MASK_DIR)

WORKING_DIRECTORY=cfg['working_folder']

ROADS=cfg['input_files']['roads']
TILES=cfg['input_files']['tiles']

os.chdir(WORKING_DIRECTORY)

# Define functions ------------------------------
def poly_from_utm(polygon, transform):
    poly_pts = []
    
    for i in np.array(polygon.exterior.coords):
        
        # Convert polygons to the image CRS
        poly_pts.append(~transform * tuple(i))
        
    # Generate a polygon object
    new_poly = Polygon(poly_pts)
    return new_poly

# Import data -----------------------------------
logger.info('Importing data...')

roads=gpd.read_file(ROADS)
tiles=gpd.read_file(TILES)

# Treat data ------------------------------------
logger.info('Making the multi-polygons for the mask...')

roads_union_geom=roads.unary_union
roads_union=gpd.GeoDataFrame({'id_roadset': [i for i in range(len(roads_union_geom.geoms))],
                            'geometry': [geo for geo in roads_union_geom.geoms]},
                            crs=roads.crs
                            )

fct_misc.test_crs(tiles.crs, roads_union.crs)
inv_masks_tiles=gpd.overlay(tiles, roads_union, how="difference")
# inv_masks.to_file(os.path.join(shp_gpkg_folder, 'test_mask.shp'))

inv_masks_tiles_3857=inv_masks_tiles.to_crs(epsg=3857)

# cf. https://lpsmlgeo.github.io/2019-09-22-binary_mask/
for tile_row in tqdm(inv_masks_tiles_3857.itertuples(), desc='Producing the masks', total=inv_masks_tiles_3857.shape[0]):
    tile_id=tile_row.id

    # Get the tile filepath
    x, y, z = tile_id.lstrip('(').rstrip(')').split(', ')
    filename=z+'_'+x+'_'+y+'.tif'
    filepath=os.path.join('obj_detector', tile_row.dataset+'-images', filename)

    # Get the tile
    with rasterio.open(os.path.join(filepath), "r") as src:
        tile_img = src.read()
        tile_meta = src.meta
    
    # fct_misc.test_crs(tile_meta['crs'], inv_masks_tiles.crs)

    im_num_bands=tile_img.shape[0]
    im_size = (tile_meta['height'], tile_meta['width'])

    try:
        polygons=[poly_from_utm(geom, src.meta['transform']) for geom in tile_row.geometry.geoms]
    except AttributeError:
        polygons=[tile_row.geometry]

    inv_mask = rasterize(shapes=polygons,
                    out_shape=im_size)
                    
    tile_mask = (1-inv_mask) * 255

    tile_img_augmented=np.ndarray(shape=(im_num_bands+1,256,256), dtype='uint8')
    tile_img_augmented[0:im_num_bands,:,:]=tile_img
    tile_img_augmented[im_num_bands,:,:]=tile_mask

    new_num_bands=tile_img_augmented.shape[0]
    mask_meta = src.meta.copy()
    if MASK_AS_BAND:
        mask_meta.update({'count': new_num_bands})
        with rasterio.open(os.path.join(filepath), 'w', **mask_meta) as dst:
            dst.write(tile_img_augmented, [band for band in range(1,tile_img_augmented.shape[0]+1)])
    else:
        mask_meta.update({'count': 1})
        with rasterio.open(os.path.join(MASK_DIR, filename), 'w', **mask_meta) as dst:
            dst.write(tile_mask, 1)