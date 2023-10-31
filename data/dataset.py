import os
import torch
import time
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
import json
import rioxarray

base_dir = os.path.dirname(os.path.abspath(__file__))

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
        self.data_interval, self.data_ids, self.shp_list, self.data_lons, self.data_lats, data_infos = self.load_raster_shp()

        # random shuffle shape file
        total_len = self.data_interval[-1] + 1
        train_len, test_len = [int(i * total_len) for i in self.config['Data']['split']]
        data_idx = np.arange(total_len)
        np.random.shuffle(data_idx)
        shp_dict = {}
        shp_dict['train'] = data_idx[:train_len]
        shp_dict['test'] = data_idx[train_len:train_len+test_len]
        shp_dict['val'] = data_idx[train_len+test_len:]

        self.shp_idx = shp_dict[split]


    def __len__(self):
        return len(self.shp_idx)

    def __getitem__(self, idx):
        index = self.shp_idx[idx]
        shp_id = np.where((self.data_interval - index) >= 0)[0][0]
        img_id = self.data_ids[shp_id]
        if shp_id > 0:
            index_rec = index - self.data_interval[shp_id - 1] - 1
        else:
            index_rec = index

        shp = self.shp_list[shp_id]
        depth = shp['Depth'].iloc[index_rec]
        depth = torch.tensor([depth], dtype=torch.float32)

        lons = np.array(self.data_lons[img_id])
        lats = np.array(self.data_lats[img_id])
        lon_idx = np.argmin(np.abs(lons - shp['Longitude'].iloc[index_rec]))
        lat_idx = np.argmin(np.abs(lats - shp['Latitude'].iloc[index_rec]))

        img_clip = self.clip_image(img_id, lon_idx, lat_idx)
        img_clip_norm = torch.from_numpy(img_clip / 255).to(torch.float32)

        outputs = {
            'image': img_clip_norm,
            'depth': depth,
            'lon': shp['Longitude'].iloc[index_rec],
            'lat': shp['Latitude'].iloc[index_rec]
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
        data_len = 0
        data_interval = []
        img_ids = []
        shp_list = []
        img_lons = []
        img_lats = []
        data_infos = []
        max_depth = 0
        min_depth = -999
        for idx, dataset in enumerate(self.dataset):
            data_spec_path = os.path.join(base_dir, '../dataset_spec', dataset +'.txt')
            with open(data_spec_path, "r") as fp:
                dataset_spec = json.load(fp)
            shp_lens = dataset_spec['shp_infos']
            #shp_names.extend(dataset_spec['shp_names'])
            for name in dataset_spec['shp_names']:
                shp_path = os.path.join(self.dataset_root, dataset, 'gt', name)
                shp = pd.read_csv(shp_path)
                if shp['Depth'].min() < max_depth:
                    max_depth = shp['Depth'].min()
                if shp['Depth'].max() > min_depth:
                    min_depth = shp['Depth'].max()
                shp_list.append(shp)
            img_lons.append(dataset_spec['lons'])
            img_lats.append(dataset_spec['lats'])
            data_infos.append(dataset_spec)
            for l in shp_lens:
                data_len += l
                data_interval.append(data_len - 1)
                img_ids.append(idx)

        print('datastet depth range from {}m to {}m \n'.format(max_depth, min_depth))

        return np.array(data_interval), img_ids, shp_list, img_lons, img_lats, data_infos

    def clip_image(self, img_id, lon_idx, lat_idx):
        """
        clip the areas on the image corresponding to the selected point
        :param img_id: image index
        :param lon_idx: select point lon coordinate in image
        :param lat_idx: select point lat coordinate in image
        :return: ndarray
        """
        dataset = self.config['Data']['dataset'][img_id]
        raster_path = os.path.join(self.dataset_root, dataset, dataset + '_color.tif')
        if not os.path.exists(raster_path):
            raster_path = os.path.join(self.dataset_root, dataset, dataset + '_color.tiff')

        raster_img = rioxarray.open_rasterio(raster_path)
        width = raster_img.shape[2]
        height = raster_img.shape[1]
        tl_y = max(0, lat_idx - self.span)
        tl_x = max(0, lon_idx - self.span)
        br_y = min(height, lat_idx + self.span + 1)
        br_x = min(width, lon_idx + self.span + 1)
        img_clip = raster_img[:, tl_y: br_y, tl_x: br_x].to_numpy()

        pad_rows = (max(self.span - lat_idx, 0), max(lat_idx + self.span + 1 - height, 0))
        pad_cols = (max(self.span - lon_idx, 0), max(lon_idx + self.span + 1 - width, 0))
        pad_dims = (0,0)

        img_clip_pad = np.pad(img_clip, (pad_dims, pad_rows, pad_cols), 'edge')

        img_clip_pad = img_clip_pad.reshape(3, -1).transpose()

        raster_img.close()

        return img_clip_pad



if __name__ == '__main__':
    import yaml
    with open('../config/config-cpu.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data = BathyDataset(config)
    sample = data.__getitem__(2)

    print('data loaded')