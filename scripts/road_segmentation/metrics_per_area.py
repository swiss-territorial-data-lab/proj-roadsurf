import os
import sys
import time
import yaml
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from math import ceil

import determine_class
sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc
from final_metrics import get_metrics

logger = fct_misc.format_logger(logger)


if __name__ == "__main__":
    tic = time.time()
    logger.info('Starting...')

    parser = ArgumentParser(description="This script generates COCO-annotated training/validation/test/other datasets for object detection tasks.")
    parser.add_argument('config_file', type=str, help='a YAML config file', default='config/config_obj_detect.yaml')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]


    FINAL_DIR = cfg['final_folder']
    FINAL_TYPES = cfg['final_types']

    os.chdir(FINAL_DIR)

    types_gdf = gpd.read_file(FINAL_TYPES)
    types_gdf = types_gdf.to_crs(2056)

    CLASSES = types_gdf.CATEGORY.unique()
    bins = [bin for bin in range(0, 2500, 25)]   #  ceil(types_gdf.area.max()), 50)]

    metrics_per_area_df = pd.DataFrame()
    for bin_nbr in tqdm(range(1, len(bins)), desc="Getting the metrics per bin of 25 m2"):
        bin_start = bins[bin_nbr-1]
        bin_end = bins[bin_nbr]
        filtered_roads_gdf = types_gdf[(types_gdf.area >= bin_start) & (types_gdf.area < bin_end)].copy()

        if filtered_roads_gdf.empty:
            continue

        _, global_metrics_df = get_metrics(filtered_roads_gdf[['CATEGORY', 'cover_type', 'tag']], CLASSES)
        global_metrics_df['bin'] = bin_start
        global_metrics_df['road_number'] = filtered_roads_gdf.shape[0] 
        metrics_per_area_df = pd.concat([metrics_per_area_df, global_metrics_df], ignore_index=True)


    fig = go.Figure()
    metrics_list = ['Pb', 'Rb', 'f1b']

    for metric in metrics_list:
        fig.add_trace(
            go.Scatter(
                x=metrics_per_area_df['bin'],
                y=metrics_per_area_df[metric],
                mode='markers+lines',
                text=metric,
                name=metric
            )
        )

    fig.update_layout(
        xaxis_title="Area [m2]",
        yaxis_title="Metrics",
        yaxis=dict(range=[0., 1])
    )

    fig.write_html(os.path.join('images', 'metrics_per_area.html'))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = metrics_per_area_df['bin'],
            y = metrics_per_area_df['road_number'],
            mode = 'markers+lines',
            text='Roads in bin',
            name='Roads per bin',
        )
    )
    fig.update_layout(
        xaxis_title="Area",
        yaxis_title="Number of roads"
    )
    fig.update_yaxes(type="log")

    fig.write_html(os.path.join('images', 'roads_per_bin.html'))