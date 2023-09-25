import rioxarray
import rasterio as rio
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)
rds = rioxarray.open_rasterio(r"C:\Users\ku500817\Desktop\bathymetry\dataset\TrainingData\clip\dalma_clip\dalma_clip.tif")
raster_lats = rds.y.to_numpy()
raster_lons = rds.x.to_numpy()

shp_path = r"C:\Users\ku500817\Desktop\bathymetry\dataset\TrainingData\clip\dalma_clip\dalma_clip.csv"
shp = pd.read_csv(shp_path)  # EPSG
idx = random.randint(0, len(shp) - 1)
i_idx = np.argmin(np.abs(raster_lats - shp['Latitude'].iloc[idx]))
j_idx = np.argmin(np.abs(raster_lons - shp['Longitude'].iloc[idx]))

print('shape latitude and longitude {},{} \n '.format(shp['Latitude'].iloc[idx], shp['Longitude'].iloc[idx]))
print('raster xarray latitude and longitude {},{} \n '.format(raster_lats[i_idx], raster_lons[j_idx]))
print('pixel idex {}, {}'.format(i_idx, j_idx))
print('done')


fig, ax = plt.subplots(figsize=(10, 5))
array = rds[:,i_idx-2: i_idx + 3,j_idx - 2: j_idx + 3]
array = array.to_numpy()
img_clip = array.reshape(3,-1).transpose()
rds[0,i_idx-2: i_idx + 3,j_idx - 2: j_idx + 3].plot(ax = ax)
ax.set_title("Visualize shp points and pixels")
ax.set_axis_off()
ax.plot(shp['Longitude'],  shp['Latitude'], 'r*')
ax.plot(shp['Longitude'].iloc[idx],  shp['Latitude'].iloc[idx], 'bo')
ax.plot(raster_lons[j_idx - 2: j_idx + 3], raster_lats[i_idx-2: i_idx + 3], 'g.')
plt.show()