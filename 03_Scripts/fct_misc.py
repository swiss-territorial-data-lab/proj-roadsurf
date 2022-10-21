import sys
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

import rasterio
from rasterio.mask import mask

import numpy as np

def test_crs(crs1, crs2 = "EPSG:2056"):
    '''
    Take the crs of two dataframes and compare them. If they are not the same, stop the script.
    '''
    if isinstance(crs1, gpd.GeoDataFrame):
        crs1=crs1.crs
    if isinstance(crs2, gpd.GeoDataFrame):
        crs2=crs2.crs

    try:
        assert(crs1 == crs2), "CRS mismatch between the two files."
    except Exception as e:
        print(e)
        sys.exit(1)

def ensure_dir_exists(dirpath):
    '''
    Test if a directory exists. If not, make it.
    '''

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"The directory {dirpath} was created.")

    return dirpath


def get_pixel_values(polygons, tile, BANDS = range(1,4), pixel_values = pd.DataFrame(), **kwargs):
    '''
    Extract the value of the raster pixels falling under the mask and save them in a dataframe.
    cf https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal

    - polygons: shapefile determining the zones where the pixels are extracted
    - tile: path to the raster image
    - BANDS: bands of the tile
    - pixel_values: dataframe to which the values for the pixels are going to be concatenated
    - kwargs: additional arguments we would like to pass the dataframe of the pixels
    '''
    
    # extract the geometry in GeoJSON format
    geoms = polygons.geometry # list of shapely geometries

    geoms = [mapping(geoms)]

    # extract the raster values values within the polygon 
    with rasterio.open(tile) as src:
        out_image, out_transform = mask(src, geoms, crop=True)

    # no data values of the original raster
    no_data=src.nodata

    if no_data is None:
        no_data=0
        # print('The value of "no data" is set to 0 by default.')
    
    for band in BANDS:

        # extract the values of the masked array
        data = out_image[band-1]

        # extract the the valid values
        val = np.extract(data != no_data, data)
        val_0 = np.extract(data == no_data, data)

        # print(f'{len(val_0)} pixels equal to the no data value ({no_data}).')

        d=pd.DataFrame({'pix_val':val, 'band_num': band, **kwargs})

        pixel_values = pd.concat([pixel_values, d],ignore_index=True)

    return pixel_values, no_data

