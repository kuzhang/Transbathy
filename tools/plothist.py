import yaml
import numpy as np
from scipy import stats
import pandas as pd
import os
import rioxarray as rxr
import json
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
NIR_thre = 500
# Loading configurations in yaml file
with open('../config/config-cpu-test.yaml', 'r') as file:
    config = yaml.safe_load(file)

dataset_root = config['Data']['data_root']
datasets = config['Data']['dataset']
shp_data = []
for dataset in datasets:
    data_path = os.path.join(dataset_root, dataset, 'gt')
    # Open the shapefile containing some in-situ data
    shp_infos = []
    shp_names = []
    raster_info = {}
    file_list = os.listdir(data_path)
    for name in file_list:
        if name.endswith('.csv'):
            shp_path = os.path.join(data_path, name)
            shp = pd.read_csv(shp_path)
            shp_dept = shp['Depth'].to_numpy()
            shp_data.extend(shp_dept)


plt.hist(shp_data, bins=100)
plt.xlabel('depth')
plt.show()