import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

import rasterio
from rasterio.mask import mask


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px




def get_pixel_values(polygons, tile, BANDS = range(1,4), pixel_values = pd.DataFrame(), **kwargs):
    '''
    Extract the value of the raster pixels falling under the mask and save them in a dataframe.

    - polygons: shapefile determining the zones where the pixels are extracted
    - tile: path to the raster image
    - BANDS: bands of the tile
    - pixel_values: dataframe to which the values for the pixels are going to be concatenated
    - kwargs: additional arguments we would like to pass the dataframe of the pixels
    '''
    
    # extract the geometry in GeoJSON format
    geoms = polygons.geometry.values # list of shapely geometries

    geoms = [mapping(geoms[0])]

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



def get_df_stats(dataframe, col, results_dict = None, to_df = False):
    '''
    Get the min, max, mean, median, std and count of a column in a dataframe and send back a dict or a dataframe

    - dataframe: dataframe from which the statistics will be calculated
    - col: sting or list of string indicating the column(s) from which the statistics will be calculated
    - result dict: dictionary for the results with the key 'min', 'max', 'mean', 'median', 'std', and 'count'
    - to_df: results from dictionary to dataframe
    '''

    if results_dict==None:
        results_dict={'min': [], 'max': [], 'mean': [], 'median': [], 'std': [], 'count': []}

    results_dict['min'].append(dataframe[col].min())
    results_dict['max'].append(dataframe[col].max())
    results_dict['mean'].append(dataframe[col].mean())
    results_dict['median'].append(dataframe[col].median())
    results_dict['std'].append(dataframe[col].std())
    results_dict['count'].append(dataframe[col].count())

    if to_df:
        results_df=pd.DataFrame(results_dict)
        return results_df
    else:
        return results_dict



def evplot(ev):
    '''
    Implementation of Kaiser's rule and the Broken stick model (MacArthur, 1957) to determine the number of components to keep in the PCA.
    https://www.mohanwugupta.com/post/broken_stick/ -> adapted for Python

    - ev: eigenvalues
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
    Determine the number of pc to keep
    '''

    pc_to_keep_kaiser=len([x for x in ev if x>sum(ev)/len(ev)])

    pc_to_keep_bsm=len([x for x in ev if x>bsm[ev.tolist().index(x)]])

    pc_to_keep=min(pc_to_keep_kaiser,pc_to_keep_bsm)

    if pc_to_keep<2:
        print(f'The number of components to keep was {pc_to_keep}. The number of components to keep is set to 1 and the number of components to plot is set to 2.')
        pc_to_keep=1
        pc_to_plot=2
    elif pc_to_keep>10:
        print(f'The number of components to keep and plot was {pc_to_keep}. It is set to a maximum limit of 10')
        pc_to_keep=10
        pc_to_plot=10
    else:
        pc_to_plot=pc_to_keep
        print(f'The number of components to keep and plot is {pc_to_keep}.')

    return pc_to_keep, pc_to_plot
        

def calculate_pca(dataset, features, to_describe,
                dirpath_tables='tables',  dirpath_images='images',
                file_pca_values='PCA_values.csv', file_pc_to_keep='PC_to_keep_evplot.jpg',
                file_graph_ind='PCA_PC1{pc}_individuals.jpg', file_graph_feat='PCA_PC1{pc}_features.jpeg',
                title_graph = 'PCA'):
    '''
    Calculate a PCA, determine the number of components to keep, plot the individuals and the variables along those components. The results as saved
    as files.

    - dataset: dataset from which the PCA will be calculated
    - features: decriptive variables of the dataset (must be numerical only)
    - to_describe: explenatory variables or the variables to describe with the PCA (FOR NOW, ONLY ONE EXPLENATORY VARIALBE CAN BE PASSED)
    - dirpath_tables: direcory for the tables
    - dirpath_images: directory for the images
    - file_pca_values: csv file where the coordoniates of the individuals after the PCA are saved
    - file_pc_to_keep: image file where the graphs for the determination of the number of principal components to keep are saved
    - file_graph_ind: image file where the graph for the individuals is saved
    - file_graph_feat: image file where the graph for the features is saveds
    - title_graph: string with the title for the graphs (the same for all)
    '''

    written_files=[]

    # 1. Define the variables and scale
    dataset.reset_index(drop=True, inplace=True)
    x=dataset.loc[:,features].values
    y=dataset.loc[:,to_describe].values

    x = StandardScaler().fit_transform(x)

    # 2. Calculate the PCA
    pca = PCA(n_components=len(features))

    coor_PC = pca.fit_transform(x)

    coor_PC_df = pd.DataFrame(data = coor_PC, columns = [f"PC{k}" for k in range(1,len(features)+1)])
    results_PCA = pd.concat([coor_PC_df, dataset[to_describe]], axis = 1)

    results_PCA.round(3).to_csv(os.path.join(dirpath_tables, file_pca_values), index=False)
    written_files.append(file_pca_values)


    # 3. Get the number of components to keep
    eigenvalues=pca.explained_variance_
    bsm, fig_pc_num = evplot(eigenvalues)

    pc_to_keep, pc_to_plot = determine_pc_num(eigenvalues, bsm)

    fig_pc_num.savefig(os.path.join(dirpath_images, file_pc_to_keep), bbox_inches='tight')
    written_files.append(file_pc_to_keep)


    # 4. Plot the graph of the individuals
    expl_var_ratio=[round(x*100,2) for x in pca.explained_variance_ratio_.tolist()]

    for pc in range(2,pc_to_plot+1):
        locals={'pc': pc}
        fig = plt.figure(figsize = (8,8))

        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(f'Principal Component 1 ({expl_var_ratio[0]}%)', fontsize = 15)
        ax.set_ylabel(f'Principal Component {pc} ({expl_var_ratio[1]}%)', fontsize = 15)
        ax.set_title(title_graph, fontsize = 20)

        targets = dataset[to_describe].unique().tolist()
        colors=[key[4:] for key in mcolors.TABLEAU_COLORS.keys()][pc_to_plot]
        for target, color in zip(targets, colors):
            indicesToKeep = results_PCA['road_type'] == target
            ax.scatter(results_PCA.loc[indicesToKeep, 'PC1']
                    , results_PCA.loc[indicesToKeep, f'PC{pc}']
                    , c = color
                    , s = 50)
        ax.legend(targets)
        ax.set_aspect(1)
        ax.grid()

        fig.savefig(os.path.join(dirpath_images, eval(f'f"{file_graph_ind}"', locals)), bbox_inches='tight')
        written_files.append(eval(f'f"{file_graph_ind}"', locals))

        # 5. Plot the graph of the variables
        labels_column=[f'Principal component {k+1} ({expl_var_ratio[k]}%)' for k in range(len(features))]
        coor_PC=pd.DataFrame(coor_PC, columns=labels_column)

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # fig = px.scatter(coor_PC, x= f'Principal component 1 ({expl_var_ratio[0]}%)', y=f'Principal component {pc} ({expl_var_ratio[1]}%)', color=results_PCA['road_type'])
        fig = px.scatter(pd.DataFrame(columns=labels_column),
                        x = f'Principal component 1 ({expl_var_ratio[0]}%)', y=f'Principal component {pc} ({expl_var_ratio[1]}%)',
                        title = title_graph)

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

        fig.update_layout(
            margin=dict(l=20, r=10, t=40, b=10),
        )

        fig.write_image(os.path.join(dirpath_images, eval(f'f"{file_graph_feat}"', locals)))
        file_graph_feat_webp=file_graph_feat.replace('jpeg','webp')
        fig.write_image(os.path.join(dirpath_images, eval(f'f"{file_graph_feat_webp}"', locals)))

        written_files.append( eval(f'f"{file_graph_feat}"', locals))
        written_files.append( eval(f'f"{file_graph_feat_webp}"', locals))

    return written_files
