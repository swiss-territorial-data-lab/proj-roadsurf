import argparse
import yaml
import os, sys
import logging, logging.config
from tqdm import tqdm

import pandas as pd
import geopandas as gpd

from rasterstats import zonal_stats
from scipy.stats import kstest

import numpy as np

import matplotlib.pyplot as plt

import fct_misc
import fct_statistics as fs

with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['statistical_analysis.py']    #  [os.path.basename(__file__)]


# Definitions of the functions

def im_of_hist_comp(band, roads, pixels_per_band, dirpath_f_images, prefix=''):
    '''
    Produce the comparison between the histogram of the pixels belonging to the road and all the pixels in its type of road cover.
    Save an image

    - band: band on which we are working
    - roads: roads of interest as a dataframe
    - pixels_per_band: dataframe with all the pixels
    - dirpath_f_images: path of the directory to save the image
    - prefix: string to insert in the file name after the indication that it contains an histogram
    '''
    written_files_fct=[]

    for road_row in roads.itertuples():
        objectid=road_row.road_id
        cover=road_row.road_type
        p_value=getattr(road_row, f'ks_p_{band}')

        road_pixels=pixels_per_band.loc[pixels_per_band['road_id'] == objectid, band]
        corresponding_pixels=pixels_per_band.loc[pixels_per_band['road_type'] == cover, band]
        other_pixels=pixels_per_band.loc[pixels_per_band['road_type'] != cover, band]

        nbr_road_pixels=road_pixels.shape[0]

        data={'pixels of the road': road_pixels, f'{cover} pixels': corresponding_pixels, 'other pixels': other_pixels}

        ks_graph=fs.compare_histograms(data,
                                    graph_title=f'''Histogram of the distribution of the {nbr_road_pixels} pixels
                                            on the {band} band (p-value: {p_value})''',
                                    axis_label='density of the pixels')

        ks_graph.savefig(os.path.join(dirpath_f_images, f'Hist_{prefix}{cover}_road_{int(objectid)}_band_{band}.jpeg'), bbox_inches='tight')

        written_files_fct.append(f'final/images/Hist_{prefix}{cover}_road_{objectid}_band_{band}.jpeg')

    plt.close('all')

    return written_files_fct


# Definition of the constants
DEBUG_MODE=cfg['debug_mode']
USE_ZONAL_STATS=cfg['use_zonal_stats']
CORRECT_BALANCE=cfg['correct_balance']

BANDS=range(1,5)
MAX_CONFIDENCE_INT = cfg['param']['max_confidence']
COUNT_THRESHOLD = cfg['param']['pixel_threshold']

DO_KS_TEST = cfg['param']['do_ks_test']
MAKE_BOXPLOTS = cfg['param']['make_boxplots']
MAKE_PCA = cfg['param']['make_pca']

PROCESSED=cfg['processed']
PROCESSED_FOLDER=PROCESSED['processed_folder']
FINAL_FOLDER=cfg['final_folder']

## Inputs
ROADS=PROCESSED_FOLDER + PROCESSED['input_files']['roads']
TILES_DIR=PROCESSED_FOLDER + PROCESSED['input_files']['images']
TILES_INFO=PROCESSED_FOLDER + PROCESSED['input_files']['tiles']

written_files=[]
dirpath_f_tables=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'tables'))
dirpath_f_images=fct_misc.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'images'))

if __name__ == "__main__":

    # Importation of the files
    roads=gpd.read_file(ROADS)
    tiles_info = gpd.read_file(TILES_INFO)


    # Data fromatting
    if DEBUG_MODE:
        tiles_info=tiles_info[1:500]
        print('Debug mode activated: only 500 tiles will be processed.')

    if False:
        unsure_roads=gpd.read_file(os.path.join(PROCESSED_FOLDER, 'shapefiles_gpkg/test_natural_roads.shp'))
        id_unsure_roads=unsure_roads['OBJECTID'].values.tolist()

        roads=roads[~roads['OBJECTID'].isin(id_unsure_roads)]

    
    if roads[roads.is_valid==False].shape[0]!=0:
       print(f"There are {roads[roads.is_valid==False].shape[0]} invalid geometries for the roads.")
       sys.exit(1)          

    simplified_roads=roads.drop(columns=['ERSTELLUNG', 'ERSTELLU_1', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M',
                'KUNSTBAUTE', 'WANDERWEGE', 'VERKEHRSBE', 'BEFAHRBARK', 'EROEFFNUNG', 'STUFE', 'RICHTUNGSG',
                'KREISEL', 'EIGENTUEME', 'VERKEHRS_1', 'NAME', 'TLM_STRASS', 'STRASSENNA', 'SHAPE_Leng'])

    roads_reproj=simplified_roads.to_crs(epsg=3857)
    tiles_info_reproj=tiles_info.to_crs(epsg=3857)

    fp_list=[]
    for tile_idx in tiles_info_reproj['id'].values:
            # Get the name of the tiles
            x, y, z = tile_idx.lstrip('(,)').rstrip('(,)').split(',')
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

    roads_base=pd.DataFrame(corrected_roads[['OBJECTID', 'BELAGSART', 'road_width', 'geometry']]).rename(columns={'OBJECTID': 'road_id', 
                            'BELAGSART':'road_type'})
    
    if USE_ZONAL_STATS:
        clipped_roads=gpd.GeoDataFrame()
        for idx in tqdm(tiles_info_reproj.index, desc='Clipping roads'):

            roads_to_tile = gpd.clip(corrected_roads, tiles_info_reproj.loc[idx,'geometry']).explode(index_parts=False)
            roads_to_tile['tile']=tiles_info_reproj.loc[idx, 'title']

            clipped_roads=pd.concat([clipped_roads,roads_to_tile], ignore_index=True)

        del roads_to_tile

    fct_misc.test_crs(corrected_roads.crs, tiles_info_reproj.crs)
    intersected_tiles=gpd.sjoin(tiles_info_reproj, corrected_roads[['OBJECTID', 'geometry']])
    intersected_tiles.drop_duplicates(subset=['id','OBJECTID'], inplace=True)

    try:
        assert not corrected_roads['OBJECTID'].duplicated().any()
    except:
        print('Some roads are separated on mulitple lines. They must be transformed to multipolygons or fused first.')
        sys.exit(1)

    pixels_per_band=pd.DataFrame()
    for road in tqdm(corrected_roads.itertuples(), total=corrected_roads.shape[0], desc='Extracting road pixels'):

        # Get the characteristics of the road
        objectid=road.OBJECTID

        # Get the corresponding tile(s)
        intersected_tiles_with_road=intersected_tiles[intersected_tiles['OBJECTID'] == objectid].copy()
        intersected_tiles_with_road.reset_index(drop=True, inplace=True)

        # Get the pixels for each tile
        pixel_values=pd.DataFrame()
        for tile_filepath in intersected_tiles_with_road['filepath'].values:
            
            pixel_values = fct_misc.get_pixel_values(road.geometry, tile_filepath, BANDS, pixel_values,
                                                    road_id=objectid)

        pixels_per_band=pd.concat([pixels_per_band, pixel_values], ignore_index=True)


    pixels_per_band=pd.merge(pixels_per_band, roads_base.drop(columns=['geometry']), on = 'road_id')

    del pixel_values
    del roads, simplified_roads, roads_reproj
    del tiles_info, tiles_info_reproj


    # Data treatment
    ## Determination of the statistics for the road segments
    print('Determination of the statistics of the roads...')

    if USE_ZONAL_STATS:
        roads_stats=pd.DataFrame()

        for tile_row in tqdm(tiles_info_reproj.itertuples(), total=tiles_info_reproj.shape[0], desc='Calculating zonal statistics'):

            roads_on_tile=clipped_roads[clipped_roads['tile']==tile_row.title]

            # Get the path of the tile
            im_path=tile_row.filepath

            roads_on_tile.reset_index(drop=True, inplace=True)

            # Calculation for each road on each band
            for road in roads_on_tile.itertuples():

                for band_num in BANDS:

                    stats=zonal_stats(road, im_path, stats=['min', 'max', 'mean', 'median','std','count'], 
                                        band=band_num, nodata=0)
                    stats_dict=stats[0]
                    stats_dict['band']=band_num
                    stats_dict['road_id']=road.OBJECTID
                    stats_dict['road_type']=road.BELAGSART
                    stats_dict['geometry']=road.geometry
                    stats_dict['tile_id']=tile_row.id

                    roads_stats = pd.concat([roads_stats, pd.DataFrame(stats_dict,index=[0])],ignore_index=True)

        del stats_dict
        del roads_on_tile

    else:
        
        roads_stats=roads_base.copy()
        for band in tqdm(BANDS, desc='Determining the stats for bands'):
            roads_stats_subset=fs.get_df_stats_groupby(pixels_per_band, f'band{band}', ['road_id'], suffix=f'_{band}')
            roads_stats_subset['road_id']=roads_stats_subset.index
            roads_stats_subset.reset_index(drop=True, inplace=True)

            roads_stats=pd.merge(roads_stats, roads_stats_subset, on='road_id')

        roads_stats['count']=roads_stats['count_1']
        roads_stats.drop(columns=[f'count_{band}' for band in BANDS], inplace=True)

        large_conf_int=sum([roads_stats[roads_stats[f'confidence_{band}'] > MAX_CONFIDENCE_INT].shape[0] for band in BANDS])
        if large_conf_int != 0:
            print(f'There are {large_conf_int} confidence intervals larger than' + 
                    f' {MAX_CONFIDENCE_INT} of pixel value.')

        del roads_stats_subset

    # roads_stats_gdf=gpd.GeoDataFrame(roads_stats)

    # dirpath=fct_misc.ensure_dir_exists(os.path.join(PROCESSED_FOLDER, 'shapefiles_gpkg'))

    # roads_stats_gdf.to_file(os.path.join(dirpath, 'roads_stats.shp'))
    # written_files.append('processed/shapefiles_gpkg/roads_stats.shp')

    roads_stats_df= roads_stats.drop(columns=['geometry'])

    dirpath=fct_misc.ensure_dir_exists(os.path.join(PROCESSED_FOLDER,'tables'))

    roads_stats_df.to_csv(os.path.join(dirpath, 'stats_roads.csv'), index=False)
    written_files.append('processed/tables/stats_roads.csv')

    roads_stats_filtered=roads_stats_df[
                                    (roads_stats_df['count'] > COUNT_THRESHOLD) 
                                    & ((roads_stats_df['confidence_1'] < MAX_CONFIDENCE_INT)
                                    | (roads_stats_df['confidence_2'] < MAX_CONFIDENCE_INT)
                                    | (roads_stats_df['confidence_3'] < MAX_CONFIDENCE_INT)
                                    | (roads_stats_df['confidence_4'] < MAX_CONFIDENCE_INT))
                                    ].drop(columns=[f'confidence_{band}' for band in BANDS]+['count'])

    print(f'{roads_stats_df.shape[0]-roads_stats_filtered.shape[0]} roads on {roads_stats_df.shape[0]}'+
            f' were dropped because they contained less than {COUNT_THRESHOLD} pixels or their confidence'+
            f' interval was higher than {MAX_CONFIDENCE_INT} on one or many bands.')

    ## Determination of the statistics for the pixels by type
    ### Get ratio between bands
    print('Calculating ratios between bands...')

    names={'1/2': 'R/G', '1/3': 'R/B', '1/4': 'R/NIR', '2/3': 'G/B', '2/4': 'G/NIR', '3/4': 'B/NIR'}
    bands_ratio=list(names.values()) + ['VgNIR-BI']

    for band in BANDS:
        for sec_band in range(band+1, max(BANDS)+1):
            pixels_per_band[names[f'{band}/{sec_band}']] = pixels_per_band[f'band{band}'].astype('float64')/pixels_per_band[f'band{sec_band}'].astype('float64')

    pixels_per_band['VgNIR-BI']=(
                pixels_per_band['band2'].astype('float64') - pixels_per_band['band4'].astype('float64'))/(
                    pixels_per_band['band2'].astype('float64') + pixels_per_band['band4'].astype('float64'))

    ### Calculate the statistics of the pixel by band and by type of road cover
    print('Calculating the statistics per band and cover...')

    cover_stats={'cover':[], 'band':[],
                'min':[], 'max':[], 'mean':[], 'median':[], 'std':[],
                'confidence': [], 'count':[]}

    for cover_type in pixels_per_band['road_type'].unique().tolist():

        for band in BANDS:
            pixels_subset=pixels_per_band[pixels_per_band['road_type']==cover_type]

            cover_stats['cover'].append(cover_type)
            cover_stats['band'].append(band)

            cover_stats=fs.get_df_stats_no_group(pixels_subset, f'band{band}', cover_stats)
    
    cover_stats['max']=[int(x) for x in cover_stats['max']] # Otherwise, the values get transformed to x-256 when converted in dataframe

    cover_stats_df=pd.DataFrame(cover_stats)

    cover_stats_df['mean']=cover_stats_df['mean'].round(1)
    cover_stats_df['std']=cover_stats_df['std'].round(1)
    cover_stats_df['confidence']=cover_stats_df['confidence'].round(1)

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

        del natural_pixels, artificial_pixels
        del natural_stats, artificial_stats

    else:
        balance=''

    ## Change the format to reader-frienldy
    print('Converting the tables to reader-friendly...')

    BANDS_STR=['red','green','blue','NIR']
    road_stats_read=roads_stats_filtered.copy()
    pixels_per_band_read=pixels_per_band.copy()

    pixels_per_band_read.rename(columns={'band1': 'red', 'band2': 'green', 'band3': 'blue', 'band4': 'NIR'}, inplace=True)

    rename_road_stats={}
    for band in BANDS:
        for stat in ['max_', 'min_', 'mean_', 'median_', 'std_', 'count_', 'confiance_']:
            rename_road_stats[stat+f'{band}']=stat+f'{BANDS_STR[band-1]}'

    road_stats_read.rename(columns=rename_road_stats, inplace=True)

    pixels_per_band_read['road_type']=pixels_per_band['road_type'].map({100: 'artificial', 200: 'natural'})
    road_stats_read.loc[:, 'road_type']=roads_stats_filtered['road_type'].map({100: 'artificial', 200: 'natural'})

    roads_stats_filtered=road_stats_read.copy()
    pixels_per_band=pixels_per_band_read.copy()


    del pixels_per_band_read
    del road_stats_read, roads_stats, roads_stats_df,       # roads_stats_gdf
    del cover_stats, cover_stats_df, large_conf_int
    

    ## Boxplots
    if MAKE_BOXPLOTS:

        print('Calculating boxplots...')

        # The green bar in the boxplot is the median
        # (cf. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.box.html)

        ### Boxplots of the pixel value
        bp_pixel_bands=pixels_per_band[BANDS_STR + ['road_type']].plot.box(by='road_type',
                                                title=f'Repartition of the values for the pixels',
                                                figsize=(10,8),
                                                grid=True)
        fig = bp_pixel_bands[0].get_figure()
        fig.savefig(os.path.join(dirpath_f_images, f'{balance}boxplot_pixel_in_bands.jpg'), bbox_inches='tight')
        written_files.append(f'final/images/{balance}boxplot_pixel_in_bands.jpg')

        bp_pixel_bands=pixels_per_band[bands_ratio  + ['road_type']].plot.box(by='road_type',
                                                title=f'Repartition of the values for the pixels',
                                                figsize=(10,8),
                                                grid=True)
        fig = bp_pixel_bands[0].get_figure()
        fig.savefig(os.path.join(dirpath_f_images, f'{balance}boxplot_pixel_in_bands_ratio.jpg'), bbox_inches='tight')
        written_files.append(f'final/images/{balance}boxplot_pixel_in_bands_ratio.jpg')


        ### Boxplots of the statistics
        for band in BANDS_STR:

            col_to_keep=[stat + f'{band}' for stat in ['max_', 'min_', 'mean_', 'median_', 'std_']]
            
            roads_stats_subset=roads_stats_filtered[col_to_keep + ['road_type']].copy()
            roads_stats_plot=roads_stats_subset.plot.box(by='road_type',
                                                        title=f'Boxplot of the statistics for the {band} band',
                                                        figsize=(30,8),
                                                        grid=True)

            fig = roads_stats_plot[0].get_figure()
            fig.savefig(os.path.join(dirpath_f_images, f'{balance}boxplot_stats_band_{band}.jpg'), bbox_inches='tight')
            written_files.append(f'final/images/{balance}boxplot_stats_band_{band}.jpg') 

        
    ## Kolmogorov-Smirnov test
    # Null-hypothesis (H_0): the two samples are drawn from the same distribution

    # https://stats.stackexchange.com/questions/81497/how-similar-are-my-2-data-sets
    # https://stats.stackexchange.com/questions/413358/how-to-test-statistics-for-the-similarity-or-dissimilarity-between-these-two-cur
    # https://www.researchgate.net/post/Is-a-two-sample-Kolmogorov-Smirnov-Test-effective-in-case-of-imbalanced-data

    if DO_KS_TEST:
        print('Executing the Kolmogorov-Smirnov test...')

        for band in BANDS_STR:
            ks=[]
            for road_row in tqdm(roads_stats_filtered.itertuples(), total=roads_stats_filtered.shape[0],
                                desc=f'Comparing road pixels to distribution on band {band}'):

                objectid=road_row.road_id
                cover=road_row.road_type

                general_values=pixels_per_band.loc[pixels_per_band['road_type']==cover, [band, 'road_id']]

                road_values=general_values.loc[general_values['road_id']==objectid, band]

                ks.append(kstest(road_values, general_values.loc[:,band]))

            ks_p_value=[float('{:0.3e}'.format(float(str(ks[k]).split(',')[1].lstrip(' pvalue=').rstrip(')')))) for k in range(len(ks))]
            roads_stats_filtered[f"ks_p_{band}"]=ks_p_value

            ks_d_value=[round(float(str(ks[k]).split(',')[0].lstrip('KstestResult(statistic=')),3) for k in range(len(ks))]
            roads_stats_filtered[f"ks_D_{band}"]=ks_d_value

        roads_stats_filtered.to_csv(os.path.join(dirpath_f_tables, 'ks_test.csv'))

        print('The null hypothesis is that the distribution for the pixels of a road is similar to'+
                ' the one of all the pixels of a road type. It is rejected when p-value < 0.05')

        dir_histograms=fct_misc.ensure_dir_exists(os.path.join(dirpath_f_images, 'histograms'))

        for band in BANDS_STR:
            
            for cover in roads_stats_filtered['road_type'].unique().tolist():

                ### Counting the results
                all_roads=roads_stats_filtered[roads_stats_filtered["road_type"] == cover][f"ks_p_{band}"].count()
                significant_roads=roads_stats_filtered[(roads_stats_filtered[f"ks_p_{band}"] > 0.05) & 
                                                    (roads_stats_filtered['road_type'] == cover)][f"ks_p_{band}"].count()
                print(f'There are {significant_roads} on {all_roads} roads with a p-value higher than 0.05'
                        + f' on band {band} with a {cover} cover.')

                ### Getting some example images
                max_ks=roads_stats_filtered[f"ks_p_{band}"].max()
                road_max_ks=roads_stats_filtered[(roads_stats_filtered[f"ks_p_{band}"] > max_ks-max_ks/100) &
                                                (roads_stats_filtered['road_type'] == cover)].reset_index(drop=True).head(5)
                written_files.extend(im_of_hist_comp(band, road_max_ks, pixels_per_band, dir_histograms, prefix='high_'))

                min_ks=roads_stats_filtered[f"ks_p_{band}"].min()
                road_min_ks=roads_stats_filtered[(roads_stats_filtered[f"ks_p_{band}"] <= min_ks+min_ks/100) & 
                                                (roads_stats_filtered['road_type'] == cover)].reset_index(drop=True).head(5)
                written_files.extend(im_of_hist_comp(band, road_min_ks, pixels_per_band, dir_histograms, prefix='low_'))

    ## PCA
    if MAKE_PCA:
        print('Calculating PCAs...') 

        ### PCA of the pixel values
        print('-- PCA of the pixel values...')

        features = BANDS_STR + bands_ratio + ['road_width']
        to_describe='road_type'

        written_files_pca_pixels=fs.calculate_pca(pixels_per_band, features, to_describe,
                    dirpath_f_tables, dirpath_f_images, 
                    file_prefix=f'{balance}PCA_pixels_',
                    title_graph='PCA for the values of the pixels on each band')

        written_files.extend(written_files_pca_pixels)

        #### PCA of the road stats
        # With separation of the bands
        print('-- PCA of the road stats (with separation of the bands...')

        for band in tqdm(BANDS_STR, desc='Processing bands'):

            features = [stat + f'{band}' for stat in ['max_', 'min_', 'mean_', 'median_', 'std_']] + ['road_width']

            to_describe='road_type'

            written_files_pca_stats=fs.calculate_pca(roads_stats_filtered, features, to_describe,
                    dirpath_f_tables, dirpath_f_images, 
                    file_prefix=f'{balance}PCA_stats_band_{band}_',
                    title_graph=f'PCA of the statistics of the roads on the {band} band')

            written_files.extend(written_files_pca_stats)


    print(f'Checkout the written files: {written_files}')