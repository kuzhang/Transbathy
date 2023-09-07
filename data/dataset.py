import os
import torch
import time
from torch.utils.data import Dataset
import numpy as np
import random
import rasterio as rio
import pandas as pd
from scipy import stats
from glob import glob
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
        idx, i_idx, j_idx = self.random_sample_points()
        img_id = self.shp['ID'].iloc[idx]

        depth = [self.shp['Depth'].iloc[idx]]
        depth = torch.tensor(depth, dtype=torch.float32)
        img_clip = self.clip_image(img_id, i_idx, j_idx)
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
        :return: lons_uniq(np.array), lats_uniq(np.array), raster_size_list(list) and shape(dataframe)
        """
        shps = []
        lons_uniq_list = []
        lats_uniq_list = []
        raster_size_list = []
        for idx, dataset in enumerate(self.dataset):
            data_path = os.path.join(self.dataset_root, dataset)
            shp_file_name = fnmatch.filter(os.listdir(data_path), '*.csv')[0]
            raster_files = fnmatch.filter(os.listdir(data_path), '*.tif')
            if len(raster_files) == 0:
                raster_files = fnmatch.filter(os.listdir(data_path), '*.tiff')
            raster_file_name = raster_files[0]

            # Open the shapefile containing some in-situ data
            shp_path = os.path.join(data_path, shp_file_name)
            shp = pd.read_csv(shp_path) # EPSG:4326-WGS 84
            shp_mod = self.remove_land_point(shp)
            shp_mod = self.remove_outlier(shp_mod)

            # Open the geotiff image file using Rasterio
            raster_path = os.path.join(data_path, raster_file_name)
            raster_img = rioxarray.open_rasterio(raster_path)
            shp_mod['ID'] = idx
            shps.append(shp_mod)

            # change loading raster image using rioxarray to fix the dataloading bugs, 7/09/2023
            #raster_lons_uniq, raster_lats_uniq, raster_size = self.create_raster_cord(raster_img)
            raster_lats = raster_img.y.to_numpy()
            raster_lons = raster_img.x.to_numpy()
            raster_size = {'width': raster_img.shape[2],
                           'height': raster_img.shape[1]}
            lons_uniq_list.append(raster_lons)
            lats_uniq_list.append(raster_lats)
            raster_size_list.append(raster_size)

        shp_concat = pd.concat(shps, axis=0, ignore_index=True)
        lons_uniq_concat = np.concatenate(lons_uniq_list, axis=0)
        lats_uniq_concat = np.concatenate(lats_uniq_list, axis=0)

        return lons_uniq_concat, lats_uniq_concat, raster_size_list, shp_concat

    def load_img(self, idx):
        """
        Load image with given index
        :param idx:
        :return: img_dn(np.array)
        """
        dataset = self.config['Data']['dataset'][idx]
        raster_path = os.path.join(self.dataset_root, dataset, dataset + '.tif')
        if not os.path.exists(raster_path):
            raster_path = os.path.join(self.dataset_root, dataset, dataset + '.tiff')

        raster_img = rio.open(raster_path)
        img_dn = raster_img.read()
        img_dn = img_dn.reshape(raster_img.height, raster_img.width, -1)  # h,w, c

        return img_dn

    def remove_land_point(self, shp):
        """
        remove the in situ data where is for land
        :param shp: dataframe
        :return: dataframe
        """
        shp_dept = shp['Depth'].to_numpy()
        if np.median(shp_dept > 0):
            shp['Depth'] = shp['Depth'] * -1
        remove_list = np.where(shp['Depth'] >= self.config['Data']['elev_threshold'])[0].tolist()
        shp = shp.drop(remove_list, axis='index')

        return shp

    def remove_outlier(self, shp):
        """
        remove the outlier in situ data
        :param shp: dataframe
        :return: dataframe
        """
        z_score = np.abs(stats.zscore(shp))
        remove_list = np.where(z_score > self.config['Data']['zscore_std'])[0].tolist()
        shp = shp.drop(shp.index[remove_list], axis='index')

        return shp

    def clip_image(self, img_id, i_idx, j_idx):
        """
        clip the areas on the image corresponding to the selected point
        :param img_id: image index
        :param i_idx: select point i coordinate in image
        :param j_idx: select point j coordinate in image
        :return: ndarray
        """
        img = self.load_img(img_id)
        img_clip = img[(j_idx - self.span): (j_idx + self.span + 1), (i_idx - self.span): (i_idx + self.span + 1), :]
        img_clip = img_clip.reshape(-1, 3)

        return img_clip

    def create_raster_cord(self, raster):
        """
        convert raster coordinates (dataframe) to numpy array and concatenate all the coordinate to an array
        :return: raster_lons_uniq(np.array), raster_lats_uniq(np.array), raster_size(dict)
        """
        width = raster.width
        height = raster.height
        raster_size={}
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        #cols = np.arange(width).tolist()
        #rows = np.arange(height).tolist()
        xs, ys = rio.transform.xy(raster.transform, rows=rows, cols=cols)
        raster_lons_uniq = np.unique(np.array(xs).flatten())  #x-coordinates
        raster_lats_uniq = np.unique(np.array(ys).flatten())  #y-coordinates
        raster_size['width'] = width
        raster_size['height'] = height

        return raster_lons_uniq, raster_lats_uniq, raster_size

    def random_sample_points(self):
        """
        random sample data points from shape file and find the matched points in image, avoid points on the edge
        :return:
        """
        sampling_done = False
        i_idx = None
        j_idx = None
        idx = None

        while not sampling_done:
            idx = random.randint(0, len(self.shp)-1)
            j_idx = np.argmin(np.abs(self.raster_lats_uniq - self.shp['Latitude'].iloc[idx]))
            i_idx = np.argmin(np.abs(self.raster_lons_uniq - self.shp['Longitude'].iloc[idx]))
            img_id = self.shp['ID'].iloc[idx]

            if i_idx in np.arange(self.span, self.raster_size[img_id]['width'] - self.span, 1) and \
                    j_idx in np.arange(self.span, self.raster_size[img_id]['height'] - self.span, 1):
                sampling_done = True
            else:
                continue

        return idx, i_idx, j_idx



if __name__ == '__main__':
    import yaml
    with open('../config/config-cpu.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data = BathyDataset(config)
    sample = data.__getitem__(0)

    print('data loaded')