import os
import torch
import time
from torch.utils.data import Dataset
import numpy as np
import random
import rasterio as rio
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from scipy import stats

def plot_2dimg(image, image_smooth):
    fig, axs = plt.subplots(2)
    axs[0].imshow(image)
    axs[1].imshow(image_smooth)
    plt.show()

def plot_3dimg(x,y,z,idx):


    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax = plt.figure(idx).add_subplot(projection='3d')
    # Plot the 3D surface
    ax.scatter(x, y, z, marker='.')
    # surf = ax.plot_surface(xv, yv, array, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    # ax.contour(xv, yv, array, zdir='z', offset=-100, cmap='coolwarm')
    # ax.contour(xv, yv, array, zdir='x', offset=-40, cmap='coolwarm')
    # ax.contour(xv, yv, array, zdir='y', offset=40, cmap='coolwarm')

    # ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
    #        xlabel='X', ylabel='Y', zlabel='Z')

    plt.show()


def load_img(config, idx):
    """
    Load image with given index
    :param idx:
    :return: img_dn(np.array)
    """
    dataset_root = config['Data']['data_root']
    dataset = config['Data']['dataset'][idx]
    raster_file_name = dataset + '.tif'
    raster_path = os.path.join(dataset_root, dataset, raster_file_name)
    raster_img = rio.open(raster_path)
    img_dn = raster_img.read()
    img_dn = img_dn.reshape(raster_img.height, raster_img.width, -1)  # h,w, c

    shp_file_name = dataset + '.csv'
    shp_path = os.path.join(dataset_root, dataset, shp_file_name)
    shp = pd.read_csv(shp_path)
    z_score = np.abs(stats.zscore(shp))
    rm_list = np.where(z_score> 2)[0].tolist()
    shp = shp.drop(rm_list, axis='index')

    row, col = np.array(raster_img.index(shp['Longitude'], shp['Latitude']))

    return img_dn, shp, row, col


with open('../config/config-cpu.yaml', 'r') as file:
    config = yaml.safe_load(file)


config['Data']['data_root'] = r'C:\Users\ku500817\Desktop\bathymetry\dataset\TrainingData'
config['Data']['dataset'] = 'Tristereo'
img, shp, row, col = load_img(config, 0)
x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
z = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)


z[row, col] = shp['Depth']
z = z.astype(np.float32)
z_smooth = cv2.medianBlur(z, 3)
depth = z[row, col]
depth_smooth = z_smooth[row, col]

dict = {
    'Longitude': shp['Longitude'],
    'Latitude': shp['Latitude'],
    'Depth': shp['Depth']
}
df = pd.DataFrame(dict)


z[z==0]=np.nan
plot_3dimg(x, y, z, 0)
z_smooth[z_smooth==0]=np.nan
plot_3dimg(x, y, z_smooth, 1)
# plot_2dimg(z, z_smooth)
print('done')
