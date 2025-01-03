import yaml
import numpy as np
from scipy import stats
import pandas as pd
import os
import rioxarray as rxr
import json


base_dir = os.path.dirname(os.path.abspath(__file__))
NIR_thre = 500
# Loading configurations in yaml file
with open('../config/config-dataprocess.yaml', 'r') as file:
    config = yaml.safe_load(file)


dataset_root = config['Data']['data_root']
datasets = config['Data']['dataset']
for dataset in datasets:
    data_path = os.path.join(dataset_root, dataset)
    shp_infos = []
    shp_names = []
    gt_paths = os.path.join(data_path, 'gt')
    for name in os.listdir(gt_paths):
        if name.endswith('.csv'):
            shp_path = os.path.join(data_path, 'gt', name)
            shp = pd.read_csv(shp_path)  # EPSG:4326-WGS 84
            shp_infos.append(len(shp))
            shp_names.append(name)
    # Open the shapefile containing some in-situ data

    raster_info = {}
    file_list = os.listdir(data_path)
    for name in file_list:
        if (name.endswith('_color.tif') or name.endswith('_color.tiff')):
            raster_path = os.path.join(data_path, name)
            raster_img = rxr.open_rasterio(raster_path)
            raster_lats = raster_img.y.to_numpy()
            raster_lons = raster_img.x.to_numpy()
            raster_info = {'width': raster_img.shape[2],
                           'height': raster_img.shape[1],
                           'lons': raster_lons.tolist(),
                           'lats': raster_lats.tolist(),
                           'name': name
                           }

    dataset_info = {'dataset': dataset,
                    'raster_name': raster_info['name'],
                    'width': raster_info['width'],
                    'height': raster_info['height'],
                    'shp_infos': shp_infos,
                    'shp_names': shp_names,
                    'lons': raster_info['lons'],
                    'lats': raster_info['lats'],
                    }

    dataspec_path = os.path.join(base_dir, '../dataset_spec', dataset + '.txt')

    with open(dataspec_path, "w") as fp:
        json.dump(dataset_info, fp)
    print("Done writing dict into {}.txt file".format(dataset))