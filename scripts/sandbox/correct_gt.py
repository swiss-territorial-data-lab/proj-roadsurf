import os
import yaml

import geopandas as gpd
import pandas as pd

with open('03_Scripts/config.yaml') as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['correct_gt.py']    #  [os.path.basename(__file__)]

INPUT = cfg['input']
INPUT_DIR =INPUT['input_folder']

ROADS = INPUT_DIR + INPUT['input_files']['initial_roads']
CORRECTIONS = INPUT_DIR + INPUT['input_files']['corrections']

roads=gpd.read_file(ROADS)
corrections=gpd.read_file(CORRECTIONS)

corrected_roads=pd.merge(roads, corrections[['OBJECTID', 'Belag_veri']], on='OBJECTID', how='left')

correct_surface=[]
for road in corrected_roads.itertuples():
    if road.Belag_veri in [100, 200, 999997, 999998]:
        correct_surface.append(int(road.Belag_veri))
    else:
        correct_surface.append(road.BELAGSART)

corrected_roads['BELAGSART']=correct_surface
corrected_roads.drop(columns=['Belag_veri'], inplace=True)

corrected_roads.to_file(os.path.join(INPUT_DIR, 'initial/TLM/Strassen/corrected_roads_inside_AOI.shp'))
