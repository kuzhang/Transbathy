import yaml
import numpy as np
from scipy import stats
import pandas as pd
import os
import rioxarray as rxr
import json

base_dir = os.path.dirname(os.path.abspath(__file__))

# Loading configurations in yaml file
with open('../config/config-cpu.yaml', 'r') as file:
    config = yaml.safe_load(file)


def remove_land_point(shp):
    """
    remove the in situ data where is for land
    :param shp: dataframe
    :return: dataframe
    """
    shp_dept = shp['Depth'].to_numpy()
    if np.median(shp_dept > 0):
        shp['Depth'] = shp['Depth'] * -1
    remove_list = np.where(shp['Depth'] >= config['Data']['elev_threshold'])[0].tolist()
    shp = shp.drop(remove_list, axis='index')
    shp.reset_index(inplace=True, drop=True)
    return shp


def remove_outlier(shp):
    """
    remove the outlier in situ data
    :param shp: dataframe
    :return: dataframe
    """
    z_score = np.abs(stats.zscore(shp))
    remove_list = np.where(z_score > config['Data']['zscore_std'])[0].tolist()
    shp = shp.drop(shp.index[remove_list], axis='index')
    shp.reset_index(inplace=True, drop=True)

    return shp


dataset_root = config['Data']['data_root']
datasets = config['Data']['dataset']
for idx, dataset in enumerate(datasets):
    data_path = os.path.join(dataset_root, dataset)
    # Open the shapefile containing some in-situ data
    shp_infos = []
    shp_names = []
    raster_info = {}
    for path, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            if name.endswith('.csv'):
                shp_path = os.path.join(data_path, name)
                shp = pd.read_csv(shp_path)  # EPSG:4326-WGS 84
                shp_mod = remove_land_point(shp)
                shp_mod = remove_outlier(shp_mod)
                output_path = os.path.join(data_path, 'gt')
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                save_file = os.path.join(output_path, name)
                shp_mod.to_csv(save_file)

                shp_infos.append(len(shp_mod))
                shp_names.append(name)

            if (name.endswith('.tif') or name.endswith('.tiff')):
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
                    'lons': raster_info['lons'],
                    'lats': raster_info['lats'],
                    'shp_infos': shp_infos,
                    'shp_names': shp_names,
                    'ID': idx
                    }

    dataspec_path = os.path.join(base_dir,'../dataset_spec', dataset + '.txt')

    with open(dataspec_path, "w") as fp:
        json.dump(dataset_info, fp)
    print("Done writing dict into {}.txt file".format(dataset))