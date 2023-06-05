import os, sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px


sys.path.insert(1, 'scripts')
from functions.fct_misc import ensure_dir_exists

def compare_histograms(data, graph_title=None, axis_label=None):
    '''
    Make histogram of density for two dataset on the same plot.

    - data: dictionnary of the data with the label as keys and the data as values
    - graph_title: title of the graph
    - axis_label: label for the y axis

    return: a matplotlib figure object with the two histograms.
    '''

    bins = np.linspace(0, 255, 55)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    for data_label in data.keys():
        ax.hist(data[data_label], bins, alpha=0.3, label=data_label, density=True)

    ax.legend(loc='upper right')
    ax.grid()

    ax.set(title=graph_title, ylabel=axis_label)

    return fig


def get_df_stats_groupby(dataframe, col, groups, suffix=''):
    '''
    Get the min, max, mean, median, std and count in a dataframe with the groupby() method and send back a dataframe

    - dataframe: dataframe from which the statistics will be calculated
    - col: sting indicating the column from which the statistics will be calculated
    - groups: list of string indicating the columns to group by
    - suffix for the name of the columns in the reuslting dataframe

    return: a dataframe with the statistics for the designated columns groupped by the designated groups.
    '''
    stats_df=dataframe.groupby(groups)[col].agg(['min', 'max', 'median', 'mean', 'count', 'std'])
    
    # Get the margin of error for > 95%
    Z = 2   # Coefficient of 1.96 rounded up
    stats_df[f'margin{suffix}'] = Z*stats_df['std']/(stats_df['count']**(1/2))

    stats_df['mean']=stats_df['mean'].round(2)
    stats_df['std']=stats_df['std'].round(2)
    stats_df[f'margin{suffix}']=stats_df[f'margin{suffix}'].round(2)

    if suffix != '':
        rename_dict={'min': f'min{suffix}', 'max': f'max{suffix}', 'median': f'median{suffix}', 'mean': f'mean{suffix}', 
                    'count': f'count{suffix}', 'std': f'std{suffix}'}
        stats_df.rename(columns=rename_dict, inplace=True)

    return stats_df

def get_df_stats_no_group(dataframe, col, results_dict = None, suffix='', to_df = False):
    '''
    Get the min, max, mean, median, std and count of a column in a dataframe and send back a dict or a dataframe

    - dataframe: dataframe from which the statistics will be calculated
    - col: sting indicating the column from which the statistics will be calculated
    - result dict: dictionary for the results with the key 'min', 'max', 'mean', 'median', 'std', 'count' and 'margin'
    - suffix for the name of the columns in the reuslting dataframe
    - to_df: results from dictionary to dataframe

    return: a dictionnary or a dataframe of the statistics for the designated column.
    '''

    if results_dict==None:
        results_dict={f'min{suffix}': [], f'max{suffix}': [], f'mean{suffix}': [], f'median{suffix}': [], f'std{suffix}': [],
                    f'count{suffix}': [], f'margin{suffix}': []}

    results_dict[f'min{suffix}'].append(int(dataframe[col].min()))
    results_dict[f'max{suffix}'].append(int(dataframe[col].max()))
    results_dict[f'mean{suffix}'].append(dataframe[col].mean().round(2))
    results_dict[f'median{suffix}'].append(dataframe[col].median())
    results_dict[f'std{suffix}'].append(dataframe[col].std().round(2))
    results_dict[f'count{suffix}'].append(dataframe[col].count())

    # Get the margin of error for > 95%
    Z = 2   # Coefficient of 1.96 rounded up
    results_dict[f'margin{suffix}'].append(np.round(Z * results_dict[f'std{suffix}'][-1] / (results_dict[f'count{suffix}'][-1]**(1/2)),
                                                        decimals=3))

    if to_df:
        results_df=pd.DataFrame(results_dict)
        return results_df
    else:
        return results_dict



def evplot(ev):
    '''
    Implementation of Kaiser's rule and the Broken stick model (MacArthur, 1957) to determine the number of components to keep in the PCA.
    cf. https://www.mohanwugupta.com/post/broken_stick/ -> adapted for Python

    - ev: eigenvalues of the PCA

    return: the list of values ofr the Broken stick model and the matplotlib figure object for both methods.
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
    Determine the number of PC to keep and to plot

    - ev: eigenvalues of the PCA
    - bsm: broken stick model as given by the function evplot

    return: the number of PC to plot 
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

    return pc_to_plot


def calculate_pca(dataset, features, to_describe, label_pc):
    '''
    Calculate a PCA

    - dataset: dataset from which the PCA will be calculated
    - features: decriptive variables of the dataset (must be numerical only)
    - to_describe: explenatory variables or the variables to describe with the PCA (FOR NOW, ONLY ONE EXPLENATORY VARIALBE CAN BE PASSED)
    - label_pc: labels for the principal components

    return: a sklearn PCA object and an array of the new coordinates.
    '''

    # 1. Define the variables and scale
    dataset.reset_index(drop=True, inplace=True)
    x=dataset.loc[:,features].values
    y=dataset.loc[:,to_describe].values

    x = StandardScaler().fit_transform(x)

    # 2. Calculate the PCA
    pca = PCA(n_components = len(features))

    coor_PC = pca.fit_transform(x)

    return pca, coor_PC


def plot_pca(coor_PC, results_PCA, pca,
            features, targets, pc_to_plot=2,
            dirpath_images='images', file_prefix='PCA_', title_graph='PCA'):
    '''
    Plot the individuals and the variables along those components. The results are saved as files.

    - coor_PC: array with the new coordinates for the principal components
    - results_PCA: dataframe with the new coordinates in the space of the PCA
    - pca: sklearn pca object
    - features: decriptive variables of the dataset (must be numerical only)
    - targets: classes of interest in the explenatory variable.
    - pc_to_plot: number of principal components to plot
    - dirpath_images: directory for the images
    - file_prefix: prefix for the names of the files that will be created
    - title_graph: string with the title for the graphs (the same for all)

    return: a list of the written files.
    '''

    written_files=[]

    expl_var_ratio=[round(x*100,2) for x in pca.explained_variance_ratio_.tolist()]
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    colors=[key[4:] for key in mcolors.TABLEAU_COLORS.keys()][:len(targets)]

    for pc in range(2,pc_to_plot+1):
        fig = plt.figure(figsize = (8,8))

        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(f'Principal Component 1 ({expl_var_ratio[0]}%)', fontsize = 15)
        ax.set_ylabel(f'Principal Component {pc} ({expl_var_ratio[1]}%)', fontsize = 15)
        ax.set_title(title_graph, fontsize = 20)

        for target, color in zip(targets, colors):
            indicesToKeep = results_PCA['road_type'] == target
            ax.scatter(results_PCA.loc[indicesToKeep, 'PC1']
                    , results_PCA.loc[indicesToKeep, f'PC{pc}']
                    , c = color
                    , s = 50)
        ax.legend(targets)
        ax.set_aspect(1)
        ax.grid()

        figpath=os.path.join(dirpath_images, file_prefix + f'PC1{pc}_individuals.jpg')
        fig.savefig(figpath, bbox_inches='tight')
        written_files.append(figpath)

        # 5. Plot the graph of the variables
        labels_column=[f'Principal component {k+1} ({expl_var_ratio[k]}%)' for k in range(len(features))]
        coor_PC=pd.DataFrame(coor_PC, columns=labels_column)

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

        file_graph_feat = os.path.join(dirpath_images, file_prefix + f'PC1{pc}_features.jpeg')
        fig.write_image(file_graph_feat)
        
        file_graph_feat_webp = file_graph_feat.replace('jpeg','webp')
        fig.write_image(file_graph_feat_webp)

        written_files.append(file_graph_feat)
        written_files.append(file_graph_feat_webp)

    return written_files


def pca_procedure(dataset, features, to_describe,
                dirpath_tables='tables',  dirpath_images='images',
                file_prefix='PCA_',
                title_graph = 'PCA'):
    '''
    Calculate a PCA (function calculate_PCA), determine the number of components to keep (function evplot and 
    determine_pc_num), save the loadings and correlations, plot the individuals and the variables along those
    components (function plot_pca).
    The results are saved as files.

    - dataset: dataset from which the PCA will be calculated
    - features: decriptive variables of the dataset (must be numerical only)
    - to_describe: explenatory variables or the variables to describe with the PCA (FOR NOW, ONLY ONE EXPLENATORY VARIALBE CAN BE PASSED)
    - dirpath_tables: direcory for the tables
    - dirpath_images: directory for the images
    - file_prefix: prefix for the names of the files that will be created
    - title_graph: string with the title for the graphs (the same for all)

    return: a list of the written files.
    '''

    written_files=[]
    _ = ensure_dir_exists(dirpath_tables)
    _ = ensure_dir_exists(dirpath_images)
    label_pc = [f'PC{x}' for x in range(1, len(features)+1)]

    file_prefix = file_prefix + '_' if file_prefix[-1]!='_' else file_prefix

    # 1 & 2. Define the variables, scale & calculate the PCA
    pca, coor_PC=calculate_pca(dataset, features, to_describe, label_pc)

    coor_PC_df = pd.DataFrame(data = coor_PC, columns = label_pc)
    results_PCA = pd.concat([coor_PC_df, dataset[to_describe]], axis = 1)

    filepath=os.path.join(dirpath_tables, file_prefix + 'values.csv')
    results_PCA.round(3).to_csv(filepath, index=False)
    written_files.append(filepath)

    # 3. Get the number of components to plot and keep
    eigenvalues=pca.explained_variance_
    bsm, fig_pc_num = evplot(eigenvalues)

    pc_to_plot = determine_pc_num(eigenvalues, bsm)

    figpath=os.path.join(dirpath_images, file_prefix + 'PC_to_keep_evplot.jpg')
    fig_pc_num.savefig(figpath, bbox_inches='tight')
    written_files.append(figpath)

    # 3 bis. Get features correlation and covariance
    # cf. https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    loading_matrix = pd.DataFrame(np.round(loadings, 2), columns = label_pc, index=features)
    filepath=os.path.join(dirpath_tables, file_prefix + 'loading_matrix.csv')
    loading_matrix.to_csv(filepath)
    written_files.append(filepath)

    corr=pd.DataFrame(np.round(np.transpose(pca.components_), 2), columns = label_pc, index = features)
    filepath=os.path.join(dirpath_tables, file_prefix + 'corr_matrix.csv')
    corr.to_csv(filepath)
    written_files.append(filepath)

    # 4 & 5. Plot the graph of the individuals and of the variables
    targets = dataset[to_describe].unique().tolist()
    
    written_files.extend(plot_pca(coor_PC, results_PCA, pca, features, targets, pc_to_plot,
                                dirpath_images, file_prefix, title_graph))

    return written_files
    
   