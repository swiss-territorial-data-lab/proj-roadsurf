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
        assert(crs1 == crs2), f"CRS mismatch between the two files ({crs1} vs {crs2})."
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
        # out_image, _ = mask(src, geoms, crop=True, filled=False)

        # no data values of the original raster
        no_data=src.nodata
    
    dico={}
    length_bands=[]
    for band in BANDS:

        # extract the values of the masked array
        data = out_image[band-1]

        # extract the the valid values
        val = np.extract(data != no_data, data)
        # val = np.extract(~data.mask, data.data)

        dico[f'band{band}']=val
        length_bands.append(len(val))

    max_length=max(length_bands)

    for band in BANDS:

        if length_bands[band-1] < max_length:

            fill=[no_data]*max_length
            dico[f'band{band}']=np.append(dico[f'band{band}'], fill[length_bands[band-1]:])

            print(f'{max_length-length_bands[band-1]} pixels was/were missing on the band {band} on the tile {tile[-18:]} and' +
                        f' got replaced with the value used of no data ({no_data}).')

    dico.update(**kwargs)
    pixels_from_tile = pd.DataFrame(dico)

    # We consider that the nodata values are where the value is 0 on each band
    if no_data is None:
        subset=pixels_from_tile[[f'band{band}' for band in BANDS]]
        pixels_from_tile = pixels_from_tile.drop(pixels_from_tile[subset.apply(lambda x: (max(x) == 0), 1)].index)

    pixel_values = pd.concat([pixel_values, pixels_from_tile],ignore_index=True)

    return pixel_values


def polygons_diff_without_artifacts(polygons, p1_idx, p2_idx):
    '''
    Make the difference of the geometry at row p2_idx with the one at the row p1_idx
    
    - polygons: dataset of polygons
    - p1_idx: index of the "obstacle" polygon in the dataset
    - p2_idx: index of the final polygon
    '''
    
    # Store intermediary results back to poly
    diff=polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']

    if diff.geom_type == 'Polygon':
        polygons.loc[p2_idx,'geometry'] -= polygons.loc[p1_idx,'geometry']

    elif diff.geom_type == 'MultiPolygon':
        # if a multipolygone is created, only keep the largest part to avoid the following error: https://github.com/geopandas/geopandas/issues/992
        polygons.loc[p2_idx,'geometry'] = max((polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']).geoms, key=lambda a: a.area)

        parts=[poly.area for poly in diff.geoms]
        parts.sort(reverse=True)
        
        for area in parts[1:]:
            if area>5:
                print(f"WARNING: when filtering for multipolygons, an area of {round(area,2)} m2 was lost for the polygon {round(polygons.loc[p2_idx,'OBJECTID'])}.")
                # To correct that, we should introduce a second id that we could prolong with the "new" road made by the multipolygon parts, while
                # maintaining the orginal id to trace the roads back at the end.
                # Or add .1, .2 ect. to the original 1 
                # Or to only have multipolygons to pass the function gpd.overlay
                # Or add id based on geometry

    return polygons


def test_valid_geom(poly_gdf, correct=False, gdf_obj_name=None):
    '''
    Test if all the geometry of a dataset are valid. When it is not the case, correct the geometries with a buffer of 0 m if correct != False
    and stop with an error otherwise.

    - poly_gdf: dataframe of geometries to check
    - correct: boolean indicating if the invalid geometries should be corrected with a buffer of 0 m
    - gdf_boj_name: name of the dataframe of the object in it to print with the error message
    '''

    try:
        assert(poly_gdf[poly_gdf.is_valid==False].shape[0]==0), \
              f"{poly_gdf[poly_gdf.is_valid==False].shape[0]} geometries are invalid{f' among {gdf_obj_name}' if gdf_obj_name else ''}."
    except Exception as e:
        print(e)
        if correct:
            print("Correction of the invalid geometries with a buffer of 0 m...")
            corrected_poly=poly_gdf.copy()
            corrected_poly.loc[corrected_poly.is_valid==False,'geometry']= \
                            corrected_poly[corrected_poly.is_valid==False]['geometry'].buffer(0)

            return corrected_poly
        else:
            sys.exit(1)

    return poly_gdf
