import os, sys
import requests

import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd

from rasterio.merge import merge
from rasterstats import zonal_stats

from glob import glob
from tqdm import tqdm
# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import fct_misc

def download_tiles(tiles_list, directory="", crs="2056", save_metadata=False, overwrite=True):
    '''
    Download geotiff tiles form a list of url
    This is a lighter version of the script XYZ.py in the helpers of the object detector.

    - tile_list: list of urls to the tif files
    - directory: directory where to save the tiles
    - crs: coordinate reference system, epsg number as string

    return: the list of the written filenames or the files already existing.
    '''

    successful_dowload=[]
    written_filenames=[]

    for tile_url in tqdm(tiles_list, desc="Downloading tiles:"):

        if not tile_url.endswith('.tif'):
            raise Exception("Filename must end with .tif")

        geotiff_filename = tile_url.split('/')[-1]
        geotiff_filepath=os.path.join(directory, geotiff_filename)

        if os.path.exists(geotiff_filepath):
            successful_dowload.append(tile_url)
            written_filenames.append(geotiff_filename)
            continue

        r = requests.get(tile_url, allow_redirects=True, verify=False)

        if r.status_code == 200:
            
            with open(geotiff_filepath, 'wb') as fp:
                fp.write(r.content)

            successful_dowload.append(tile_url)
            written_filenames.append(geotiff_filename)

    if set(tiles_list)!=set(successful_dowload):
        print("Some file were not successfully downloaded.")
        print('Missing files:')
        for file in tiles_list:
            if file not in successful_dowload:
                print(file)
        sys.exit(1)

    return written_filenames


def make_mosaic(directory, files_list=None, out_filepath=None):
    '''
    Making a mosaic from a list of raster files.
    cf. https://automating-gis-processes.github.io/CSC18/lessons/L6/raster-mosaic.html

    - directory: directory where to find the files
    - files_list: list of the files name
    - out_filepath: drectory where to savec the mosaic
    - crs: coordinate reference system as a string number

    return: path to the written file.
    '''

    if files_list==None:
        filepaths=glob(os.path.join(directory, '*.tif'))
    else:
        filepaths=[os.path.join(directory, file) for file in files_list]

    if out_filepath==None:
        out_filepath=os.path.join(directory, 'mosaic.tif')
    elif not out_filepath.endswith('.tif'):
        raise Exception("Filename must end with .tif")

    if len(filepaths)==0:
        raise Exception( "No file found to do the mosaic.")

    src_files_to_mosaic = []

    for fp in filepaths:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    
    out_meta = src.meta.copy()
    out_crs=src.crs

    out_meta.update({"driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "crs": out_crs
                    })

    with rasterio.open(out_filepath, "w", **out_meta) as dest:
        dest.write(mosaic)

    return out_filepath


if __name__ == '__main__':

    GENERATE_MOSAIC=False
    CALCULATE_ZONAL_STATS=True

    INITIAL_FOLDER="data/DEM"
    PROCESSED_FOLDER="outputs"
    DEM_PROCESSED_FOLDER=fct_misc.ensure_dir_exists(os.path.join(PROCESSED_FOLDER, 'DEM'))
    FINAL_FOLDER="outputs/final"

    URL_FILE="ch.swisstopo.swissalti3d-nZXjr9Tu_res2m.csv"
    ROADS_POLYGONS="shp_gpkg/types_from_detections.shp"


    if GENERATE_MOSAIC:
        files_url=pd.read_csv(os.path.join(INITIAL_FOLDER, URL_FILE), header=None)
        filenames=download_tiles(files_url[0].unique().tolist(), INITIAL_FOLDER)

        print('Generating the mosaic from tiles...')
        mosaic_path=make_mosaic(INITIAL_FOLDER, filenames, os.path.join(DEM_PROCESSED_FOLDER, "DEM_aoi.tif"))
        
        print(f'The mosaic for the DEM was written at the following location:{mosaic_path}')

    
    if CALCULATE_ZONAL_STATS:
        print('Calculating zonal stats...')

        if not GENERATE_MOSAIC:
            mosaic_path=os.path.join(DEM_PROCESSED_FOLDER, "DEM_aoi.tif")

        print('-- Reading files...')
        with rasterio.open(mosaic_path) as src:
            dem_array=src.read(1)
            affine=src.transform
            dem_crs=src.crs

        labels=gpd.read_file(os.path.join(FINAL_FOLDER, ROADS_POLYGONS))

        print('-- Calculating zonal stats...')
        
        labels.reset_index(drop=True, inplace=True)
        labels.to_crs(crs=dem_crs, inplace=True)
        fct_misc.test_crs(dem_crs, labels)

        zs_df=pd.DataFrame(zonal_stats(labels, dem_array, affine=affine,
                                        stats=['min', 'max', 'mean', 'median', 'std'], nodata=-9999))
        zs_roads=pd.concat([labels, zs_df], axis=1)
        
        print('-- Exporting file...')
        path_zs = os.path.join(PROCESSED_FOLDER, 'shapefiles_gpkg/roads_dem_zs.shp')
        zs_roads.to_file(path_zs)
        print(f'The file for the zonal stats was written at the following location: {path_zs}')