import rioxarray
import rasterio as rio
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# compare rioxarray loading and rasterio loading
random.seed(0)
def create_raster_cord(raster):
    """
    convert raster coordinates (dataframe) to numpy array and concatenate all the coordinate to an array
    :return: raster_lons_uniq(np.array), raster_lats_uniq(np.array), raster_size(dict)
    """
    width = raster.width
    height = raster.height
    raster_size = {}
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    # cols = np.arange(width).tolist()
    # rows = np.arange(height).tolist()
    xs, ys = rio.transform.xy(raster.transform, rows=rows, cols=cols)
    raster_lons_uniq = np.unique(np.array(xs).flatten())  # x-coordinates
    raster_lats_uniq = np.unique(np.array(ys).flatten())  # y-coordinates
    raster_size['width'] = width
    raster_size['height'] = height

    return raster_lons_uniq, raster_lats_uniq
rds = rioxarray.open_rasterio(r"C:\Users\ku500817\Desktop\bathymetry\dataset\From Fahim\insitu\clip\dalma_clip\dalma_clip.tif")
raster_lats = rds.y.to_numpy()
raster_lons = rds.x.to_numpy()

# raster_img = rio.open(r"C:\Users\ku500817\Desktop\bathymetry\dataset\From Fahim\insitu\clip\dalma_clip\dalma_clip.tif")
# raster_lons_uniq, raster_lats_uniq= create_raster_cord(raster_img)

shp_path = r"C:\Users\ku500817\Desktop\bathymetry\dataset\From Fahim\insitu\clip\dalma_clip\dalma_clip.csv"
shp = pd.read_csv(shp_path)  # EPSG
idx = random.randint(0, len(shp) - 1)
# i_idx = np.argmin(np.abs(raster_lats_uniq - shp['Latitude'].iloc[idx]))
# j_idx = np.argmin(np.abs(raster_lons_uniq - shp['Longitude'].iloc[idx]))
# print('pixel idex {}, {}'.format(i_idx, j_idx))
i_idx = np.argmin(np.abs(raster_lats - shp['Latitude'].iloc[idx]))
j_idx = np.argmin(np.abs(raster_lons - shp['Longitude'].iloc[idx]))
print('pixel idex {}, {}'.format(i_idx, j_idx))

print('shape latitude and longitude {},{} \n '.format(shp['Latitude'].iloc[idx], shp['Longitude'].iloc[idx]))
# print('raster image latitude and longitude {},{} \n '.format(raster_lats_uniq[i_idx], raster_lons_uniq[j_idx]))
print('raster xarray latitude and longitude {},{} \n '.format(raster_lats[i_idx], raster_lons[j_idx]))
print('pixel idex {}, {}'.format(i_idx, j_idx))
print('done')


fig, ax = plt.subplots(figsize=(10, 5))
array = rds[:,i_idx-2: i_idx + 3,j_idx - 2: j_idx + 3]
array = array.to_numpy()
img_clip = array.reshape(3,-1).transpose()
rds[0,i_idx-2: i_idx + 3,j_idx - 2: j_idx + 3].plot(ax = ax)
ax.set_title("Lidar Digital Elevation Model (DEM) \n Boulder Flood 2013")
ax.set_axis_off()
ax.plot(shp['Longitude'],  shp['Latitude'], 'r*')
ax.plot(shp['Longitude'].iloc[idx],  shp['Latitude'].iloc[idx], 'bo')
ax.plot(raster_lons[j_idx - 2: j_idx + 3], raster_lats[i_idx-2: i_idx + 3], 'g.')
plt.show()