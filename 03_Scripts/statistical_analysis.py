import argparse
import yaml
import os, sys
import logging, logging.config
from tqdm import tqdm

import pandas as pd
import geopandas as gpd

from rasterstats import zonal_stats
from shapely.geometry.multipolygon import MultiPolygon

import numpy as np

import fct_misc
import fct_statistics as fs

with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['statistical_analysis.py']    #  [os.path.basename(__file__)]


# Definitions of the functions


# Definition of the constants
DEBUG_MODE=cfg['debug_mode']
USE_ZONAL_STATS=cfg['use_zonal_stats']
CORRECT_BALANCE=cfg['correct_balance']

BANDS=range(1,5)
MAX_CONFIDANCE_INT=cfg['param']['max_confidance']
COUNT_THRESHOLD = cfg['param']['threshold']

PROCESSED=cfg['processed']
PROCESSED_FOLDER=PROCESSED['processed_folder']
FINAL_FOLDER=cfg['final_folder']

## Inputs
ROADS=PROCESSED_FOLDER + PROCESSED['input_files']['roads']
TILES_DIR=PROCESSED_FOLDER + PROCESSED['input_files']['images']
TILES_INFO=PROCESSED_FOLDER + PROCESSED['input_files']['tiles']

written_files=[]

if __name__ == "__main__":

    # Importation of the files
    roads=gpd.read_file(ROADS)
    tiles_info = gpd.read_file(TILES_INFO)


    # Data treatment
    if DEBUG_MODE:
        tiles_info=tiles_info[1:500]
        print('Debug mode activated: only 500 tiles will be processed.')
    
    if roads[roads.is_valid==False].shape[0]!=0:
       print(f"There are {roads[roads.is_valid==False].shape[0]} invalid geometries for the roads.")
       sys.exit(1)          

    simplified_roads=roads.drop(columns=['ERSTELLUNG', 'ERSTELLU_1', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M',
                'KUNSTBAUTE', 'WANDERWEGE', 'VERKEHRSBE', 'BEFAHRBARK', 'EROEFFNUNG', 'STUFE', 'RICHTUNGSG',
                'KREISEL', 'EIGENTUEME', 'VERKEHRS_1', 'NAME', 'TLM_STRASS', 'STRASSENNA', 'SHAPE_Leng', 'Width'])

    roads_reproj=simplified_roads.to_crs(epsg=3857)
    tiles_info_reproj=tiles_info.to_crs(epsg=3857)

    fp_list=[]
    for tile_idx in tiles_info_reproj.index:
            # Get the name of the tiles
            x, y, z = tiles_info_reproj.loc[tile_idx,'id'].lstrip('(,)').rstrip('(,)').split(',')
            im_name = z.lstrip() + '_' + x + '_' + y.lstrip() + '.tif'
            im_path = os.path.join(TILES_DIR, im_name)
            fp_list.append(im_path)

    tiles_info_reproj['filepath']=fp_list

    fct_misc.test_crs(roads_reproj.crs, tiles_info_reproj.crs)

    if roads_reproj[roads_reproj.is_valid==False].shape[0]!=0:
       print(f"There are {roads_reproj[roads_reproj.is_valid==False].shape[0]} invalid geometries for the roads after the reprojection.")

       print("Correction of the roads presenting an invalid geometry with a buffer of 0 m...")
       corrected_roads=roads_reproj.copy()
       corrected_roads.loc[corrected_roads.is_valid==False,'geometry']=corrected_roads[corrected_roads.is_valid==False]['geometry'].buffer(0)

    clipped_roads=gpd.GeoDataFrame()
    for idx in tqdm(tiles_info_reproj.index, desc='Clipping roads'):

        roads_to_tile = gpd.clip(corrected_roads, tiles_info_reproj.loc[idx,'geometry']).explode(index_parts=False)
        roads_to_tile['tile']=tiles_info_reproj.loc[idx, 'title']

        clipped_roads=pd.concat([clipped_roads,roads_to_tile], ignore_index=True)



    ## Determination of the statistics for the road segments
    print('Determination of the statistics of the roads...')

    if USE_ZONAL_STATS:
        roads_stats=pd.DataFrame()

        for tile_idx in tqdm(tiles_info_reproj.index, desc='Calculating zonal statistics'):

            roads_on_tile=clipped_roads[clipped_roads['tile']==tiles_info_reproj.loc[tile_idx,'title']]

            # Get the path of the tile
            im_path=tiles_info_reproj.loc[tile_idx,'filepath']

            roads_on_tile.reset_index(drop=True, inplace=True)

            # Calculation for each road on each band
            for road_idx in roads_on_tile.index:

                road=roads_on_tile.iloc[road_idx:road_idx+1]

                if road.shape[0]>1:
                    print('More than one road is being tested.')
                    sys.exit(1)

                for band_num in BANDS:

                    stats=zonal_stats(road, im_path, stats=['min', 'max', 'mean', 'median','std','count'], 
                                        band=band_num, nodata=0)
                    stats_dict=stats[0]
                    stats_dict['band']=band_num
                    stats_dict['road_id']=road.loc[road_idx,'OBJECTID']
                    stats_dict['road_type']=road.loc[road_idx,'BELAGSART']
                    stats_dict['geometry']=road.loc[road_idx,'geometry']
                    stats_dict['tile_id']=tiles_info_reproj.loc[tile_idx,'id']

                    roads_stats = pd.concat([roads_stats, pd.DataFrame(stats_dict,index=[0])],ignore_index=True)

    else:
        roads_stats={'band':[], 'road_id': [], 'road_type': [], 'geometry': [],
                    'min':[], 'max':[], 'mean':[], 'median':[], 'std':[], 'count':[], 'confidance': []}

        for road_idx in tqdm(corrected_roads.index, desc='Extracting road statistics'):

            # Get the characteristics of the road
            objectid=corrected_roads.loc[road_idx, 'OBJECTID']
            cover_type=corrected_roads.loc[road_idx, 'BELAGSART']
            road=corrected_roads.loc[corrected_roads['OBJECTID'] == objectid,['OBJECTID', 'BELAGSART', 'geometry']]
            road.reset_index(inplace=True, drop=True)
            geometry = road.loc[0,'geometry'] if road.shape[0]==1 else MultiPolygon([road.loc[k,'geometry'] for k in road.index])

            if objectid in roads_stats['road_id']:
                continue
            
            # Get the corresponding tile(s)
            fct_misc.test_crs(road.crs, tiles_info_reproj.crs)
            intersected_tiles=gpd.overlay(tiles_info_reproj, road)

            intersected_tiles.drop_duplicates(subset=['id'], inplace=True)
            intersected_tiles.reset_index(drop=True, inplace=True)

            pixel_values=pd.DataFrame()

            # Get the pixels for each tile
            for tile_idx in intersected_tiles.index:

                # Get the name of the tiles
                im_path = intersected_tiles.loc[tile_idx,'filepath']
                
                pixel_values, no_data = fs.get_pixel_values(road, im_path, BANDS, pixel_values,
                                                            road_id=objectid, road_cover=cover_type)

            if pixel_values.empty:
                continue

            for band in BANDS:
                pixels_subset=pixel_values[pixel_values['band_num']==band]

                roads_stats['band'].append(band)
                roads_stats['road_id'].append(objectid)
                roads_stats['road_type'].append(cover_type)
                roads_stats['geometry'].append(geometry)

                roads_stats=fs.get_df_stats(pixels_subset, 'pix_val', roads_stats)

        roads_stats['max']=[int(x) for x in roads_stats['max']]
        roads_stats['min']=[int(x) for x in roads_stats['min']]

        roads_stats=pd.DataFrame(roads_stats)

    large_conf_int=roads_stats[roads_stats['confidance'] > MAX_CONFIDANCE_INT]
    if not large_conf_int.empty:
        print(f'There are {large_conf_int.shape[0]} roads with a confidance interval larger than {MAX_CONFIDANCE_INT} of pixel value')

    if not roads_stats[roads_stats['std'] == 0].empty:
        roads_stats_corr = roads_stats[roads_stats['std'] != 0].copy()
        print(f'''{roads_stats.shape[0]-roads_stats_corr.shape[0]} road(s) was/were dropped, because of a standard deviation of 0.
                OBJECTID: {roads_stats.loc[roads_stats['std'] == 0, "road_id"].values.tolist()}''')

        roads_stats=roads_stats_corr.copy()

    roads_stats['mean']=roads_stats['mean'].round(2)
    roads_stats['std']=roads_stats['std'].round(2)
    roads_stats['confidance']=roads_stats['confidance'].round(2)

    roads_stats_gdf=gpd.GeoDataFrame(roads_stats)

    dirpath=fct_misc.ensure_dir_exists(os.path.join(PROCESSED_FOLDER, 'shapefiles_gpkg'))

    # roads_stats_gdf.to_file(os.path.join(dirpath, 'roads_stats.shp'))
    # written_files.append('processed/shapefiles_gpkg/roads_stats.shp')

    roads_stats_df= roads_stats.drop(columns=['geometry'])

    dirpath=fct_misc.ensure_dir_exists(os.path.join(PROCESSED_FOLDER,'tables'))

    roads_stats_df.to_csv(os.path.join(dirpath, 'stats_roads.csv'), index=False)
    written_files.append('processed/tables/stats_roads.csv')

    roads_stats_filtered=roads_stats_df[(roads_stats_df['count'] > COUNT_THRESHOLD) 
                                        & (roads_stats_df['confidance'] < MAX_CONFIDANCE_INT)]

    print(f'{roads_stats_df.shape[0]-roads_stats_filtered.shape[0]} roads on {roads_stats_df.shape[0]}'+
            f' were dropped because they contained less than {COUNT_THRESHOLD} pixels or their confidance'+
            f' interval was higher than {MAX_CONFIDANCE_INT}.')


    ## Determination of the statistics for the pixels by type

    ### Create a table with the values of pixels on a road
    # cf https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal

    pixel_values=pd.DataFrame()

    for tile_idx in tqdm(tiles_info_reproj.index, desc='Getting pixel values'):

        roads_on_tile=clipped_roads[clipped_roads['tile']==tiles_info_reproj.loc[tile_idx,'title']]
        tile = tiles_info_reproj.loc[tile_idx,'filepath']

        for cover_type in roads_on_tile['BELAGSART'].unique().tolist():

            road_shapes=roads_on_tile[roads_on_tile['BELAGSART']==cover_type]

            pixel_values, no_data =fs.get_pixel_values(road_shapes, tile, BANDS, pixel_values, road_type=cover_type)


    ### Create a new table with a column per band (just reformatting the table)
    pixels_per_band={'road_type':[], 'band1':[], 'band2':[], 'band3':[], 'band4':[]}

    for cover_type in pixel_values['road_type'].unique().tolist():

        for band in BANDS:

            pixels_list=pixel_values.loc[(pixel_values['road_type']==cover_type) & (pixel_values['band_num']==band),
                ['pix_val']]['pix_val'].to_list()

            pixels_per_band[f'band{band}'].extend(pixels_list)

        # Following part to change. Probably, better handling of the no data would avoid this mistake
        max_pixels=max(len(pixels_per_band['band1']), len(pixels_per_band['band2']), 
                        len(pixels_per_band['band3']), len(pixels_per_band['band4']))

        for band in BANDS:
            len_pixels_serie=len(pixels_per_band[f'band{band}'])

            if len_pixels_serie!=max_pixels:

                fill=[no_data]*max_pixels
                pixels_per_band[f'band{band}'].extend(fill[len_pixels_serie:])

                print(f'{max_pixels-len_pixels_serie} pixels were missing on the band {band} for the road cover {cover_type}.' +
                        f' There were replaced with the value used of no data ({no_data})')


        pixels_per_band['road_type'].extend([cover_type]*len(pixels_list))

    pixels_per_band=pd.DataFrame(pixels_per_band)

    ### Get ratio between bands
    names={'1/2': 'R/G', '1/3': 'R/B', '1/4': 'R/NIR', '2/3': 'G/B', '2/4': 'G/NIR', '3/4': 'B/NIR'}
    bands_ratio=list(names.values())

    for band in BANDS:
        for sec_band in range(band+1, max(BANDS)+1):
            pixels_per_band[names[f'{band}/{sec_band}']] = pixels_per_band[f'band{band}']/pixels_per_band[f'band{sec_band}']


    ### Calculate the statistics of the pixel by band and by type of road cover

    cover_stats={'cover':[], 'band':[],
                'min':[], 'max':[], 'mean':[], 'median':[], 'std':[],
                'confidance': [], 'iq25':[], 'iq75':[], 'count':[]}

    for cover_type in pixel_values['road_type'].unique().tolist():

        for band in BANDS:
            pixels_subset=pixel_values[(pixel_values['band_num']==band) & (pixel_values['road_type']==cover_type)]

            cover_stats['cover'].append(cover_type)
            cover_stats['band'].append(band)

            cover_stats=fs.get_df_stats(pixels_subset, 'pix_val', cover_stats)
            cover_stats['iq25'].append(pixels_subset['pix_val'].quantile(.25))
            cover_stats['iq75'].append(pixels_subset['pix_val'].quantile(.75))
    
    cover_stats['max']=[int(x) for x in cover_stats['max']] # Otherwise, the values get transformed to x-256 when converted in dataframe

    cover_stats_df=pd.DataFrame(cover_stats)

    large_conf_int=cover_stats_df[cover_stats_df['confidance'] > MAX_CONFIDANCE_INT]
    if not large_conf_int.empty:
        print(f'''There are {large_conf_int.shape[0]} roads with a confidance interval larger than 
                {MAX_CONFIDANCE_INT} of pixel value''')

    cover_stats_df['mean']=cover_stats_df['mean'].round(1)
    cover_stats_df['std']=cover_stats_df['std'].round(1)
    cover_stats_df['confidance']=cover_stats_df['confidance'].round(1)

    dirpath=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'tables') )

    cover_stats_df.to_csv(os.path.join(dirpath, 'statistics_roads_by_type.csv'), index=False)
    written_files.append('final/tables/statistics_roads_by_type.csv')

    if CORRECT_BALANCE:
        print('Taking only a subset of the artifical roads and pixels to have a balanced dataset.')

        natural_pixels=pixels_per_band[pixels_per_band['road_type']==200]
        natural_stats=roads_stats_filtered[roads_stats_filtered['road_type']==200]

        artificial_pixels=pixels_per_band[pixels_per_band['road_type']==100].reset_index(drop=True)
        artificial_stats=roads_stats_filtered[roads_stats_filtered['road_type']==100].reset_index(drop=True)

        artificial_pixels_subset=artificial_pixels.sample(frac=natural_pixels.shape[0]/artificial_pixels.shape[0],
                                                        random_state=1)
        artificial_stats_subset=artificial_stats.sample(frac=natural_stats.shape[0]/artificial_stats.shape[0],
                                                        random_state=9)

        pixels_per_band=pd.concat([artificial_pixels_subset, natural_pixels], ignore_index=True)
        roads_stats_filtered=pd.concat([artificial_stats_subset,natural_stats], ignore_index=True)

        balance='balanced_'

    else:
        balance=''

    ## Change the format to reader-frienldy
    BANDS_STR=['Red','Green','Blue','NIR']
    road_stats_read=roads_stats_filtered.copy()
    pixels_per_band_read=pixels_per_band.copy()

    pixels_per_band_read.rename(columns={'band1': 'Red', 'band2': 'Green', 'band3': 'Blue', 'band4': 'NIR'}, inplace=True)
    road_stats_read.loc[:, 'band']=roads_stats_filtered['band'].map({1: 'Red', 2: 'Green', 3: 'Blue', 4: 'NIR'})

    pixels_per_band_read['road_type']=pixels_per_band['road_type'].map({100: 'artificial', 200: 'natural'})
    road_stats_read.loc[:, 'road_type']=roads_stats_filtered['road_type'].map({100: 'artificial', 200: 'natural'})

    roads_stats_filtered=road_stats_read.copy()
    pixels_per_band=pixels_per_band_read.copy()

    ## Boxplots 
    print('Calculating boxplots...')

    dirpath_images=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'images'))

    # The green bar in the boxplot is the median
    # (cf. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.box.html)

    ### Boxplots of the pixel value
    bp_pixel_bands=pixels_per_band[BANDS_STR + ['road_type']].plot.box(by='road_type',
                                            title=f'Repartition of the values for the pixels',
                                            figsize=(10,8),
                                            grid=True)
    fig = bp_pixel_bands[0].get_figure()
    fig.savefig(os.path.join(dirpath_images, f'{balance}boxplot_pixel_in_bands.jpg'))
    written_files.append(f'final/images/{balance}boxplot_pixel_in_bands.jpg')

    bp_pixel_bands=pixels_per_band[bands_ratio  + ['road_type']].plot.box(by='road_type',
                                            title=f'Repartition of the values for the pixels',
                                            figsize=(10,8),
                                            grid=True)
    fig = bp_pixel_bands[0].get_figure()
    fig.savefig(os.path.join(dirpath_images, f'{balance}boxplot_pixel_in_bands_ratio.jpg'))
    written_files.append(f'final/images/{balance}boxplot_pixel_in_bands_ratio.jpg')


    ### Boxplots of the statistics
    for band in BANDS_STR:
        roads_stats_subset=roads_stats_filtered[roads_stats_filtered['band']==band].drop(columns=['count', 'band', 'road_id'])
        roads_stats_plot=roads_stats_subset.plot.box(by='road_type',
                                                    title=f'Boxplot of the statistics for the band {band}',
                                                    figsize=(30,8),
                                                    grid=True)

        fig = roads_stats_plot[0].get_figure()
        fig.savefig(os.path.join(dirpath_images, f'{balance}boxplot_stats_band_{band}.jpg'))
        written_files.append(f'final/images/{balance}boxplot_stats_band_{band}.jpg') 

    

    ## PCA
    print('Calculating PCAs...') 

    ### PCA of the pixel values
    print('-- PCA of the pixel values...')

    features = BANDS_STR + bands_ratio
    to_describe='road_type'

    dirpath_tables=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'tables'))

    written_files_pca_pixels=fs.calculate_pca(pixels_per_band, features, to_describe,
                dirpath_tables, dirpath_images, 
                file_prefix=f'{balance}PCA_pixels_',
                title_graph='PCA for the values of the pixels on each band')

    written_files.extend(written_files_pca_pixels)

    #### PCA of the road stats
    # With separation of the bands
    print('-- PCA of the road stats (with separation of the bands...')

    for band in tqdm(BANDS_STR, desc='Processing bands'):
        roads_stats_filtered_subset=roads_stats_filtered[roads_stats_filtered['band']==band]

        roads_stats_filtered_subset.reset_index(drop=True, inplace=True)
        features = ['min', 'max', 'mean', 'std', 'median']

        to_describe='road_type'

        written_files_pca_stats=fs.calculate_pca(roads_stats_filtered_subset, features, to_describe,
                dirpath_tables, dirpath_images, 
                file_prefix=f'{balance}PCA_stats_band_{band}_',
                title_graph=f'PCA of the statistics of the roads on the {band} band')

        written_files.extend(written_files_pca_stats)

    print(f'Checkout the written files: {written_files}')