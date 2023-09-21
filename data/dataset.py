import os
import torch
import time
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
import fnmatch
import rioxarray

class BathyDataset(Dataset):

    def __init__(self, config, split='train'):
        """

        :param config: user defined model parameters, yaml file
        :param split: dataset split
        """
        self.seed(config['Seed'])
        self.config = config
        self.dataset_root = self.config['Data']['data_root']
        self.dataset = self.config['Data']['dataset']
        self.span = self.config['Data']['span']
        self.raster_lons_uniq, self.raster_lats_uniq, self.raster_size, shp = self.load_raster_shp()

        # random shuffle shape file
        shp_shuffle = shp.sample(frac=1)
        train_len, test_len = [int(i * shp.shape[0]) for i in self.config['Data']['split']]
        shp_dict = {}
        shp_dict['train'] = shp_shuffle.iloc[:train_len,:]
        shp_dict['test'] = shp_shuffle.iloc[train_len:train_len+test_len,:]
        shp_dict['val'] = shp_shuffle.iloc[train_len+test_len:, :]

        self.shp = shp_dict[split]


    def __len__(self):
        return len(self.shp)

    def __getitem__(self, index):
        idx, lon_idx, lat_idx = self.random_sample_points()
        img_id = self.shp['ID'].iloc[idx]
        print('load shape file:{}'.format(self.shp['name'].iloc[idx]))

        depth = [self.shp['Depth'].iloc[idx]]
        depth = torch.tensor(depth, dtype=torch.float32)
        img_clip = self.clip_image(img_id, lon_idx, lat_idx)
        img_clip_norm = torch.from_numpy(img_clip / 255).to(torch.float32)

        outputs = {
            'image': img_clip_norm,
            'depth': depth
        }

        return outputs


    def seed(self, seed_value):
        """ Seed
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)


    def load_raster_shp(self):
        """
        load raster cordinates and shape file
        :return: lons_uniq(list), lats_uniq(list), raster_size_list(list) and shape(dataframe)
        """
        shps = []
        lons_uniq_list = []
        lats_uniq_list = []
        raster_size_list = []
        for idx, dataset in enumerate(self.dataset):
            data_path = os.path.join(self.dataset_root, dataset)

            # Open the shapefile containing some in-situ data
            shp_files = fnmatch.filter(os.listdir(data_path), '*.csv')
            for shp_file_name in shp_files:
                shp_path = os.path.join(data_path, shp_file_name)
                shp = pd.read_csv(shp_path) # EPSG:4326-WGS 84
                shp_mod = self.remove_land_point(shp)
                shp_mod = self.remove_outlier(shp_mod)
                shp_mod['ID'] = idx
                shp_mod['name'] = shp_file_name
                shps.append(shp_mod)

            # Open the geotiff image file using Rasterio
            raster_path = os.path.join(self.dataset_root, dataset, dataset + '.tif')
            if not os.path.exists(raster_path):
                raster_path = os.path.join(self.dataset_root, dataset, dataset + '.tiff')
            raster_img = rioxarray.open_rasterio(raster_path)

            raster_lats = raster_img.y.to_numpy()
            raster_lons = raster_img.x.to_numpy()
            raster_size = {'width': raster_img.shape[2],
                           'height': raster_img.shape[1]}
            lons_uniq_list.append(raster_lons)
            lats_uniq_list.append(raster_lats)
            raster_size_list.append(raster_size)

        shp_concat = pd.concat(shps, axis=0, ignore_index=True)

        return lons_uniq_list, lats_uniq_list, raster_size_list, shp_concat

    def clip_image(self, img_id, lon_idx, lat_idx):
        """
        clip the areas on the image corresponding to the selected point
        :param img_id: image index
        :param lon_idx: select point lon coordinate in image
        :param lat_idx: select point lat coordinate in image
        :return: ndarray
        """
        dataset = self.config['Data']['dataset'][img_id]
        raster_path = os.path.join(self.dataset_root, dataset, dataset + '.tif')
        if not os.path.exists(raster_path):
            raster_path = os.path.join(self.dataset_root, dataset, dataset + '.tiff')
        print('load raster file:{}'.format(dataset))

        raster_img = rioxarray.open_rasterio(raster_path)
        img_clip = raster_img[:, (lat_idx - self.span): (lat_idx + self.span + 1), (lon_idx - self.span): (lon_idx + self.span + 1)].to_numpy()
        img_clip = img_clip.reshape(3, -1).transpose()

        return img_clip

    def random_sample_points(self):
        """
        random sample data points from shape file and find the matched points in image, avoid points on the edge
        :return:
        """
        lat_idx = None
        lon_idx = None
        idx = None

        while lon_idx is None or lat_idx is None:
            idx = random.randint(0, len(self.shp)-1)
            img_id = self.shp['ID'].iloc[idx]
            lat_idx = np.argmin(np.abs(self.raster_lats_uniq[img_id] - self.shp['Latitude'].iloc[idx]))
            lon_idx = np.argmin(np.abs(self.raster_lons_uniq[img_id] - self.shp['Longitude'].iloc[idx]))

            if lon_idx in np.arange(self.span, self.raster_size[img_id]['width'] - self.span, 1) and \
                    lat_idx in np.arange(self.span, self.raster_size[img_id]['height'] - self.span, 1):
                break
            else:
                continue

        return idx, lon_idx, lat_idx



if __name__ == '__main__':
    import yaml
    with open('../config/config-cpu.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data = BathyDataset(config)
    sample = data.__getitem__(0)

    print('data loaded')