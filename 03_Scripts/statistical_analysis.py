import argparse
import yaml
import os, sys
import logging, logging.config
from tqdm import tqdm

import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from shapely.geometry import mapping

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import plotly.express as px

import misc_fct

with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['statistical_analysis.py']    #  [os.path.basename(__file__)]


# Definitions of the functions

def evplot(ev):
    '''
    Implementation of Kaiser's rule and the Broken stick model (MacArthur, 1957) to determine the number of components to keep in the PCA.
    https://www.mohanwugupta.com/post/broken_stick/ -> adapted for Python
    '''

    n=len(ev)

    # Broken stick model (MacArthur 1957)
    j=np.arange(n)+1
    bsm=[1/n]
    for k in range(n-1):
        bsm.append(bsm[k] + 1/(n-1-k))
    bsm=[100*x/n for x in bsm]
    bsm.reverse()

    avg_ev=sum(ev)/len(ev)

    # Plot figures
    fig = plt.figure(figsize = (8,8))

    ax = fig.add_subplot(2,1,1)
    bx = fig.add_subplot(2,1,2)

    ## Kaiser rule
    ax.bar(j,ev)
    ax.axhline(y=avg_ev, color='r', linestyle='-')

    ## Broken stick model
    bx.bar(j-0.25, ev, color='y', width=0.5)
    bx.bar(j+0.25, bsm, color='r', width=0.5)

    return bsm, fig
    

def determine_pc_num(ev, bsm):
    '''
    Determine the number of principal components to keep and to plot based on 
    the minimum of the Kaiser rule and the broken stick model.
    '''

    pc_to_keep_kaiser=len([x for x in ev if x>sum(ev)/len(ev)])

    pc_to_keep_bsm=len([x for x in ev if x>bsm[ev.tolist().index(x)]])

    pc_to_keep=min(pc_to_keep_kaiser,pc_to_keep_bsm)

    if pc_to_keep<2:
        print(f'Number of components to keep was {pc_to_keep}. Number of components to keep set to 1 and number of components to plot set to 2.')
        pc_to_keep=1
        pc_to_plot=2
    else:
        pc_to_plot=pc_to_keep
        print(f'Number of components to keep and plot is {pc_to_keep}.')

    return pc_to_keep, pc_to_plot
        


# Definition of the constants
DEBUG_MODE=cfg['debug_mode']
CORRECT_BALANCE=cfg['correct_balance']
BANDS=range(1,5)
COUNT_THRESHOLD = 50

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
        tiles_info=tiles_info[1:1000]
    
    if roads[roads.is_valid==False].shape[0]!=0:
       print(f"There are {roads[roads.is_valid==False].shape[0]} invalid geometries for the road.")
       sys.exit(1)          

    simplified_roads=roads.drop(columns=['ERSTELLUNG', 'ERSTELLU_1', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M','KUNSTBAUTE', 'WANDERWEGE',
                'VERKEHRSBE', 'BEFAHRBARK', 'EROEFFNUNG', 'STUFE', 'RICHTUNGSG', 'KREISEL', 'EIGENTUEME', 'VERKEHRS_1', 'NAME', 'TLM_STRASS', 'STRASSENNA', 
                'SHAPE_Leng', 'Width'])

    roads_reproj=simplified_roads.to_crs(epsg=3857)
    tiles_info_reproj=tiles_info.to_crs(epsg=3857)
    misc_fct.test_crs(roads_reproj.crs, tiles_info_reproj.crs)

    if roads_reproj[roads_reproj.is_valid==False].shape[0]!=0:
       print(f"There are {roads_reproj[roads_reproj.is_valid==False].shape[0]} invalid geometries for the road after the reprojection.")

       print("Correction of the roads presenting an invalid geometry with a buffer...")
       corrected_roads=roads_reproj.copy()
       corrected_roads.loc[corrected_roads.is_valid==False,'geometry']=corrected_roads[corrected_roads.is_valid==False]['geometry'].buffer(0)

    clipped_roads=gpd.GeoDataFrame()
    for idx in tqdm(tiles_info_reproj.index, desc='Clipping roads'):

        roads_to_tile = gpd.clip(corrected_roads, tiles_info_reproj.loc[idx,'geometry']).explode(index_parts=False)
        roads_to_tile['tile']=tiles_info_reproj.loc[idx, 'title']

        clipped_roads=pd.concat([clipped_roads,roads_to_tile], ignore_index=True)



    ## Determination of the statistics for the road segments
    roads_stats=pd.DataFrame()
    fp_list=[]

    for tile_idx in tqdm(tiles_info_reproj.index, desc='Calculating zonal statistics'):

        roads_on_tile=clipped_roads[clipped_roads['tile']==tiles_info_reproj.loc[tile_idx,'title']]

        # Get the name of the tiles
        x, y, z = tiles_info_reproj.loc[tile_idx,'id'].lstrip('(,)').rstrip('(,)').split(',')
        im_name = z.lstrip() + '_' + x + '_' + y.lstrip() + '.tif'
        im_path = os.path.join(TILES_DIR, im_name)
        fp_list.append(im_path)

        roads_on_tile.reset_index(drop=True, inplace=True)

        # Calculation for each road on each band
        for road_idx in roads_on_tile.index:

            road=roads_on_tile.iloc[road_idx:road_idx+1]

            if road.shape[0]>1:
                print('More than one road is being tested.')
                sys.exit(1)

            for band_num in BANDS:

                stats=zonal_stats(road, im_path, stats=['min', 'max', 'mean', 'median','std','count'], band=band_num, nodata=0)
                stats_dict=stats[0]
                stats_dict['band']=band_num
                stats_dict['road_id']=road.loc[road_idx,'OBJECTID']
                stats_dict['road_type']=road.loc[road_idx,'BELAGSART']
                stats_dict['geometry']=road.loc[road_idx,'geometry']
                stats_dict['tile_id']=tiles_info_reproj.loc[tile_idx,'id']

                roads_stats = pd.concat([roads_stats, pd.DataFrame(stats_dict,index=[0])],ignore_index=True)

    roads_stats['mean']=roads_stats['mean'].round(2)
    roads_stats['std']=roads_stats['std'].round(2)

    tiles_info_reproj['filepath']=fp_list

    roads_stats_gdf=gpd.GeoDataFrame(roads_stats)

    dirpath=misc_fct.ensure_dir_exists(os.path.join(PROCESSED_FOLDER, 'shapefiles_gpkg'))

    # roads_stats_gdf.to_file(os.path.join(dirpath, 'roads_stats.shp'))
    # written_files.append('processed/shapefiles_gpkg/roads_stats.shp')

    roads_stats_df= roads_stats.drop(columns=['geometry'])

    dirpath=misc_fct.ensure_dir_exists(os.path.join(PROCESSED_FOLDER,'tables'))

    roads_stats_df.to_csv(os.path.join(dirpath, 'stats_roads.csv'), index=False)
    written_files.append('processed/tables/stats_roads.csv')

    roads_stats_filtered=roads_stats_df[roads_stats_df['count']>COUNT_THRESHOLD]

    print(f"{roads_stats_df.shape[0]-roads_stats_filtered.shape[0]} roads on {roads_stats_df.shape[0]} were dropped because they contained less than {COUNT_THRESHOLD} pixels.")



    ## Determination of the statistics for the pixels by type

    ### Create a table with the values of pixels on a road
    # cf https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal

    pixel_values=pd.DataFrame()

    for tile_idx in tqdm(tiles_info_reproj.index, desc='Getting pixel values'):

        roads_on_tile=clipped_roads[clipped_roads['tile']==tiles_info_reproj.loc[tile_idx,'title']]
        dataset = tiles_info_reproj.loc[tile_idx,'filepath']

        for cover_type in roads_on_tile['BELAGSART'].unique().tolist():

            road_shapes=roads_on_tile[roads_on_tile['BELAGSART']==cover_type]

            # extract the geometry in GeoJSON format
            geoms = road_shapes.geometry.values # list of shapely geometries

            geoms = [mapping(geoms[0])]

            # extract the raster values values within the polygon 
            with rasterio.open(dataset) as src:
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

                d=pd.DataFrame({'pix_val':val, 'band_num': band, 'road_type': cover_type})

                pixel_values = pd.concat([pixel_values, d],ignore_index=True)


    ### Create a new table with a column per band (just reformatting the table)
    pixels_per_band={'road_type':[], 'band1':[], 'band2':[], 'band3':[], 'band4':[]}

    for cover_type in pixel_values['road_type'].unique().tolist():

        for band in BANDS:

            pixels_list=pixel_values.loc[(pixel_values['road_type']==cover_type) & (pixel_values['band_num']==band), ['pix_val']]['pix_val'].to_list()
            pixels_per_band[f'band{band}'].extend(pixels_list)

        # Following part to change. Probably, better handling of the no data would avoid this mistake
        max_pixels=max(len(pixels_per_band['band1']), len(pixels_per_band['band2']), len(pixels_per_band['band3']), len(pixels_per_band['band4']))

        for band in BANDS:
            len_pixels_serie=len(pixels_per_band[f'band{band}'])

            if len_pixels_serie!=max_pixels:

                fill=[no_data]*max_pixels
                pixels_per_band[f'band{band}'].extend(fill[len_pixels_serie:])

                print(f'{max_pixels-len_pixels_serie} pixels where missing on the band {band} for the road cover {cover_type}. There where replaced with the value used of no data ({no_data})')


        pixels_per_band['road_type'].extend([cover_type]*len(pixels_list))

    pixels_per_band=pd.DataFrame(pixels_per_band)


    ### Calculate the statistics of the pixel by band and by type of road cover

    cover_stats={'cover':[], 'band':[], 'min':[], 'max':[], 'mean':[], 'median':[], 'std':[], 'iq25':[], 'iq75':[], 'count':[]}

    for cover_type in pixel_values['road_type'].unique().tolist():

        for band in BANDS:
            pixels_subset=pixel_values[(pixel_values['band_num']==band) & (pixel_values['road_type']==cover_type)]

            cover_stats['cover'].append(cover_type)
            cover_stats['band'].append(band)
            cover_stats['min'].append(pixels_subset['pix_val'].min())
            cover_stats['max'].append(pixels_subset['pix_val'].max())
            cover_stats['mean'].append(pixels_subset['pix_val'].mean())
            cover_stats['median'].append(pixels_subset['pix_val'].median())
            cover_stats['std'].append(pixels_subset['pix_val'].std())
            cover_stats['iq25'].append(pixels_subset['pix_val'].quantile(.25))
            cover_stats['iq75'].append(pixels_subset['pix_val'].quantile(.75))
            cover_stats['count'].append(pixels_subset['pix_val'].count())
    
    cover_stats['max']=[int(x) for x in cover_stats['max']] # Otherwise, the values get transformed to x-256 when converted in dataframe

    cover_stats_df=pd.DataFrame(cover_stats)
    cover_stats_df['mean']=cover_stats_df['mean'].round(1)
    cover_stats_df['std']=cover_stats_df['std'].round(1)

    dirpath=misc_fct.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'tables') )

    cover_stats_df.to_csv(os.path.join(dirpath, 'statistics_roads_by_type.csv'), index=False)
    written_files.append('final/tables/statistics_roads_by_type.csv')

    if CORRECT_BALANCE:
        print('Taking only a subset of the artifical roads and pixels to have a balanced dataset.')

        natural_pixels=pixels_per_band[pixels_per_band['road_type']==200]
        natural_stats=roads_stats_filtered[roads_stats_filtered['road_type']==200]

        artificial_pixels=pixels_per_band[pixels_per_band['road_type']==100].reset_index(drop=True)
        artificial_stats=roads_stats_filtered[roads_stats_filtered['road_type']==100].reset_index(drop=True)

        artificial_pixels_subset=artificial_pixels.sample(frac=natural_pixels.shape[0]/artificial_pixels.shape[0], random_state=1)
        artificial_stats_subset=artificial_stats.sample(frac=natural_stats.shape[0]/artificial_stats.shape[0], random_state=9)

        pixels_per_band=pd.concat([artificial_pixels_subset, natural_pixels], ignore_index=True)
        roads_stats_filtered=pd.concat([artificial_stats_subset,natural_stats], ignore_index=True)

        balance='_balanced'

    else:
        balance=''

    ## Change the format to reader-frienldy
    pixels_per_band.rename(columns={'band1': 'NIR', 'band2': 'Red', 'band3': 'Green', 'band4': 'Blue'}, inplace=True)
    roads_stats_filtered['band']=roads_stats_filtered['band'].replace({1: 'NIR', 2: 'R', 3: 'G', 4: 'B'})
    BANDS=['NIR','R','G','B']

    pixels_per_band['road_type']=pixels_per_band['road_type'].replace({100: 'artificial', 200: 'natural'})
    roads_stats_filtered['road_type']=roads_stats_filtered['road_type'].replace({100: 'artificial', 200: 'natural'})

    ## Boxplots 
    print('Calculating boxplots...')

    dirpath_images=misc_fct.ensure_dir_exists(os.path.join(FINAL_FOLDER, 'images'))

    # The green bar in the boxplot is the median (cf. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.box.html)

    ### Boxplots of the pixel value
    bp_pixel_bands=pixels_per_band.plot.box(by='road_type', title=f'Repartition of the values for the pixels', figsize=(10,8), grid=True)
    fig = bp_pixel_bands[0].get_figure()
    fig.savefig(os.path.join(dirpath_images, f'boxplot_pixel_in_bands{balance}.jpg'))
    written_files.append(f'final/images/boxplot_pixel_in_bands{balance}.jpg')


    ### Boxplots of the statistics
    for band in BANDS:
        roads_stats_subset=roads_stats_filtered[roads_stats_filtered['band']==band].drop(columns=['count', 'band', 'road_id'])
        roads_stats_plot=roads_stats_subset.plot.box(by='road_type', figsize=(30,8), title=f'Boxplot of the statistics for the band {band}', grid=True)

        fig = roads_stats_plot[0].get_figure()
        fig.savefig(os.path.join(dirpath_images, f'boxplot_stats_band_{band}{balance}.jpg'))
        written_files.append(f'final/images/boxplot_stats_band_{band}{balance}.jpg') 

    

    ## PCA
    # cf. https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    print('Calculating PCAs...') 

    ### PCA of the pixel values
    print('-- PCA of the pixel values...')
    #### 1. Define the variables and scale

    pixels_per_band.reset_index(drop=True, inplace=True)
    features = ['NIR', 'R', 'G', 'B']

    x = pixels_per_band.loc[:, features].values
    y = pixels_per_band.loc[:,['road_type']].values

    x = StandardScaler().fit_transform(x)

    #### 2. Calculate the PCA
    pca = PCA(n_components=len(features))

    coor_PC = pca.fit_transform(x)

    coor_PC_df = pd.DataFrame(data = coor_PC, columns = [f"PC{k}" for k in range(1,len(features)+1)])
    results_PCA = pd.concat([coor_PC_df, pixels_per_band[['road_type']]], axis = 1)

    results_PCA.to_csv(os.path.join(FINAL_FOLDER, f'tables/PCA_pixel_values{balance}.csv'), index=False)
    written_files.append(f'final/tables/PCA_pixel_values{balance}.csv')

    #### 3. Get the number of components to keep
    eigenvalues=pca.explained_variance_
    bsm, fig_pc_num = evplot(eigenvalues)

    pc_to_keep, pc_to_plot = determine_pc_num(eigenvalues, bsm)

    fig_pc_num.savefig(os.path.join(dirpath_images, f'PCA_pixels_PC_to_keep_evplot{balance}.jpg'))
    written_files.append(f'final/images/PCA_pixels_PC_to_keep_evplot{balance}.jpg')

    #### 4. Plot the graph of the individuals
    expl_var_ratio=[round(x*100,2) for x in pca.explained_variance_ratio_.tolist()]

    for pc in range(2,pc_to_plot+1):
        fig = plt.figure(figsize = (8,8))

        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(f'Principal Component 1 ({expl_var_ratio[0]}%)', fontsize = 15)
        ax.set_ylabel(f'Principal Component {pc} ({expl_var_ratio[1]}%)', fontsize = 15)
        ax.set_title('PCA for the values of the pixels on each band', fontsize = 20)

        targets = pixels_per_band['road_type'].unique().tolist()
        colors = ['r', 'b']
        for target, color in zip(targets,colors):
            indicesToKeep = results_PCA['road_type'] == target
            ax.scatter(results_PCA.loc[indicesToKeep, 'PC1']
                    , results_PCA.loc[indicesToKeep, f'PC{pc}']
                    , c = color
                    , s = 50)
        ax.legend(targets)
        ax.set_aspect(1)
        ax.grid()

        fig.savefig(os.path.join(dirpath_images, f'PCA_pixels_PC1{pc}_individuals{balance}.jpg'))
        written_files.append(f'final/images/PCA_pixels_PC1{pc}_individuals{balance}.jpg')

    #### 5. Plot the graph of the variables
    labels_column=[f'Principal component {k+1} ({expl_var_ratio[k]}%)' for k in range(len(features))]
    coor_PC=pd.DataFrame(coor_PC, columns=labels_column)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # fig = px.scatter(coor_PC, x= f'Principal component 1 ({expl_var_ratio[0]}%)', y=f'Principal component 2 ({expl_var_ratio[1]}%)', color=results_PCA['road_type'])
    fig=px.scatter(pd.DataFrame(columns=labels_column),x= f'Principal component 1 ({expl_var_ratio[0]}%)', y=f'Principal component 2 ({expl_var_ratio[1]}%)')

    for i, feature in enumerate(features):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )

        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )

    fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    )

    fig.write_image(os.path.join(dirpath_images,f'PCA_pixels_PC12_features{balance}.jpeg'))
    fig.write_image(os.path.join(dirpath_images,f'PCA_pixels_PC12_features{balance}.webp'))

    written_files.append(f'final/images/PCA_pixels_PC12_features{balance}.jpeg')
    written_files.append(f'final/images/PCA_pixels_PC12_features{balance}.webp')


    #### PCA of the road stats
    # With separation of the bands
    print('-- PCA of the road stats (with separation of the bands...')
    for band in tqdm(BANDS, desc='Processing bands'):
        roads_stats_filtered_subset=roads_stats_filtered[roads_stats_filtered['band']==band]

        roads_stats_filtered_subset.reset_index(drop=True, inplace=True)
        features = ['min', 'max', 'mean', 'std','median']

        # Separating out the features
        x = roads_stats_filtered_subset.loc[:, features].values

        # Separating out the target
        y = roads_stats_filtered_subset.loc[:,['road_type']].values

        # Standardizing the features
        x = StandardScaler().fit_transform(x)

        pca = PCA(n_components=len(features))

        coor_PC = pca.fit_transform(x)

        coor_PC_df = pd.DataFrame(data = coor_PC, columns = [f"PC{k}" for k in range(1,len(features)+1)])
        results_PCA = pd.concat([coor_PC_df, roads_stats_filtered_subset[['road_type']]], axis = 1)

        results_PCA.to_csv(os.path.join(FINAL_FOLDER, f'tables/PCA_stats{band}{balance}.csv'))
        written_files.append(f'final/tables/PCA_stats{band}{balance}.csv')

        eigenvalues=pca.explained_variance_
        bsm, fig_pc_num = evplot(eigenvalues)

        pc_to_keep, pc_to_plot = determine_pc_num(eigenvalues, bsm)

        fig_pc_num.savefig(os.path.join(dirpath_images, f'PCA_stats{band}_PC_to_keep_evplot{balance}.jpg'))
        written_files.append(f'final/images/PCA_stats{band}_PC_to_keep_evplot{balance}.jpg')

        expl_var_ratio=[round(x*100,2) for x in pca.explained_variance_ratio_.tolist()]

        labels_column=[f'Principal component {k+1} ({expl_var_ratio[k]}%)' for k in range(len(features))]

        for pc in range(2,pc_to_plot+1):
            fig = plt.figure(figsize = (8,8))

            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel(f'Principal Component 1 ({expl_var_ratio[0]}%)', fontsize = 15)
            ax.set_ylabel(f'Principal Component {pc} ({expl_var_ratio[1]}%)', fontsize = 15)
            ax.set_title('PCA for the road statistics on each band', fontsize = 20)

            targets = roads_stats_filtered_subset['road_type'].unique().tolist()
            colors = ['r', 'b']
            for target, color in zip(targets,colors):
                indicesToKeep = results_PCA['road_type'] == target
                ax.scatter(results_PCA.loc[indicesToKeep, 'PC1']
                        , results_PCA.loc[indicesToKeep, f'PC{pc}']
                        , c = color
                        , s = 50)
            ax.legend(targets)
            ax.set_aspect(1)
            ax.grid()

            fig.savefig(os.path.join(dirpath_images, f'PCA_stats{band}_PC1{pc}_individuals{balance}.jpg'))
            written_files.append(f'final/images/PCA_stats{band}_PC1{pc}_individuals{balance}.jpg')

            coor_PC=pd.DataFrame(coor_PC, columns=labels_column)

            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            # fig = px.scatter(coor_PC, x= f'Principal component 1 ({expl_var_ratio[0]}%)', y=f'Principal component {pc} ({expl_var_ratio[1]}%)', color=results_PCA['road_type'])
            fig = px.scatter(pd.DataFrame(columns=labels_column),x= f'Principal component 1 ({expl_var_ratio[0]}%)', y=f'Principal component 2 ({expl_var_ratio[1]}%)')

            for i, feature in enumerate(features):
                fig.add_shape(
                    type='line',
                    x0=0, y0=0,
                    x1=loadings[i, 0],
                    y1=loadings[i, 1]
                )

                fig.add_annotation(
                    x=loadings[i, 0],
                    y=loadings[i, 1],
                    ax=0, ay=0,
                    xanchor="center",
                    yanchor="bottom",
                    text=feature,
                )

            fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
            )

        fig.write_image(os.path.join(dirpath_images,f'PCA_stats{band}_PC1{pc}_features{balance}.jpeg'))
        fig.write_image(os.path.join(dirpath_images,f'PCA_stats{band}_PC1{pc}_features{balance}.webp'))

        written_files.append(f'final/images/PCA_stats{band}_PC1{pc}_features{balance}.jpeg')
        written_files.append(f'final/images/PCA_stats{band}_PC1{pc}_features{balance}.webp')