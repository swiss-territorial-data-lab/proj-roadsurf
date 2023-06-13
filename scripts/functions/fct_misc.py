import os, sys
import logging, logging.config

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

import rasterio
from rasterio.mask import mask

import numpy as np

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('XYZ')

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

    return: the path to the verified directory.
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

    return: a dataframe with the pixel values on each band and the keyworded arguments.
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

            logger.warning(f'{max_length-length_bands[band-1]} pixels was/were missing on the band {band} on the tile {tile[-18:]} and' +
                        f' got replaced with the value used of no data ({no_data}).')

    dico.update(**kwargs)
    pixels_from_tile = pd.DataFrame(dico)

    # We consider that the nodata values are where the value is 0 on each band
    if no_data is None:
        subset=pixels_from_tile[[f'band{band}' for band in BANDS]]
        pixels_from_tile = pixels_from_tile.drop(pixels_from_tile[subset.apply(lambda x: (max(x) == 0), 1)].index)

    pixel_values = pd.concat([pixel_values, pixels_from_tile],ignore_index=True)

    return pixel_values


def polygons_diff_without_artifacts(polygons, p1_idx, p2_idx, keep_everything=False):
    '''
    Make the difference of the geometry at row p2_idx with the one at the row p1_idx
    
    - polygons: dataframe of polygons
    - p1_idx: index of the "obstacle" polygon in the dataset
    - p2_idx: index of the final polygon
    - keep_everything: boolean indicating if we should keep large parts that would be eliminated otherwise

    return: a dataframe of the polygons where the part of p1_idx overlapping with p2_idx has been erased. The parts of
    multipolygons can be all kept or just the largest one (longer process).
    '''
    
    # Store intermediary results back to poly
    diff=polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']

    if diff.geom_type == 'Polygon':
        polygons.loc[p2_idx,'geometry'] -= polygons.loc[p1_idx,'geometry']

    elif diff.geom_type == 'MultiPolygon':
        # if a multipolygone is created, only keep the largest part to avoid the following error: https://github.com/geopandas/geopandas/issues/992
        polygons.loc[p2_idx,'geometry'] = max((polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']).geoms, key=lambda a: a.area)

        # The threshold to which we consider that subparts are still important is hard-coded at 10 units.
        limit=10
        parts_geom=[poly for poly in diff.geoms if poly.area>limit]
        if len(parts_geom)>1 and keep_everything:
            parts_area=[poly.area for poly in diff.geoms if poly.area>limit]
            parts=pd.DataFrame({'geometry':parts_geom,'area':parts_area})
            parts.sort_values(by='area', ascending=False, inplace=True)
            
            new_row_serie=polygons.loc[p2_idx].copy()
            new_row_dict={'OBJECTID': [], 'OBJEKTART': [], 'KUNSTBAUTE': [], 'BELAGSART': [], 'geometry': [], 
                        'GDB-Code': [], 'Width': [], 'saved_geom': []}
            new_poly=0
            for elem_geom in parts['geometry'].values[1:]:
                
                new_row_dict['OBJECTID'].append(int(str(int(new_row_serie.OBJECTID))+str(new_poly)))
                new_row_dict['geometry'].append(elem_geom)
                new_row_dict['OBJEKTART'].append(new_row_serie.OBJEKTART)
                new_row_dict['KUNSTBAUTE'].append(new_row_serie.KUNSTBAUTE)
                new_row_dict['BELAGSART'].append(new_row_serie.BELAGSART)
                new_row_dict['GDB-Code'].append(new_row_serie['GDB-Code'])
                new_row_dict['Width'].append(new_row_serie.Width)
                new_row_dict['saved_geom'].append(new_row_serie.saved_geom)

                new_poly+=1

            polygons=pd.concat([polygons, pd.DataFrame(new_row_dict)], ignore_index=True)

    return polygons


def test_valid_geom(poly_gdf, correct=False, gdf_obj_name=None):
    '''
    Test if all the geometry of a dataset are valid. When it is not the case, correct the geometries with a buffer of 0 m
    if correct != False and stop with an error otherwise.

    - poly_gdf: dataframe of geometries to check
    - correct: boolean indicating if the invalid geometries should be corrected with a buffer of 0 m
    - gdf_boj_name: name of the dataframe of the object in it to print with the error message

    return: a dataframe with only valid geometries.
    '''

    try:
        assert(poly_gdf[poly_gdf.is_valid==False].shape[0]==0), \
            f"{poly_gdf[poly_gdf.is_valid==False].shape[0]} geometries are invalid {f' among the {gdf_obj_name}' if gdf_obj_name else ''}."
    except Exception as e:
        logger.error(e)
        if correct:
            logger.warning("Correction of the invalid geometries with a buffer of 0 m...")
            corrected_poly=poly_gdf.copy()
            corrected_poly.loc[corrected_poly.is_valid==False,'geometry']= \
                            corrected_poly[corrected_poly.is_valid==False]['geometry'].buffer(0)

            return corrected_poly
        else:
            sys.exit(1)

    logger.info(f"There aren't any invalid geometries{f' among the {gdf_obj_name}' if gdf_obj_name else ''}.")

    return poly_gdf