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


def get_pixel_values(geoms, tile, BANDS = range(1,4), pixel_values = pd.DataFrame(), **kwargs):
    '''
    Extract the value of the raster pixels falling under the mask and save them in a dataframe.
    cf https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal

    - geoms: list of shapely geometries determining the zones where the pixels are extracted
    - tile: path to the raster image
    - BANDS: bands of the tile
    - pixel_values: dataframe to which the values for the pixels are going to be concatenated
    - kwargs: additional arguments we would like to pass the dataframe of the pixels
    '''
    
    # extract the geometry in GeoJSON format
    geoms = [mapping(geoms)]

    # extract the raster values values within the polygon 
    with rasterio.open(tile) as src:
        out_image, _ = mask(src, geoms, crop=True)

        # no data values of the original raster
        no_data=src.nodata
    
    dico={}
    length_bands=[]
    for band in BANDS:

        # extract the values of the masked array
        data = out_image[band-1]

        # extract the the valid values
        val = np.extract(data != no_data, data)

        dico[f'band{band}']=val
        length_bands.append(len(val))

    max_length=max(length_bands)

    for band in BANDS:

        if length_bands[band-1] < max_length:

            fill=[no_data]*max_length
            dico[f'band{band}']=np.append(dico[f'band{band}'], fill[length_bands[band-1]:])

            print(f'{max_length-length_bands[band-1]} pixels was/were missing on the band {band} on the tile {tile} and' +
                        f' got replaced with the value used of no data ({no_data}).')

    dico.update(**kwargs)
    pixels_from_tile = pd.DataFrame(dico)

    if no_data is None:
        subset=pixels_from_tile[[f'band{band}' for band in BANDS]]
        pixels_from_tile = pixels_from_tile.drop(pixels_from_tile[subset.apply(lambda x: (max(x) == 0), 1)].index)

    pixel_values = pd.concat([pixel_values, pixels_from_tile],ignore_index=True)

    return pixel_values

