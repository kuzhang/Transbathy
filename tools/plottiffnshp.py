import rioxarray
import rasterio as rio
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)
rds = rioxarray.open_rasterio(r"C:\Users\ku500817\Desktop\bathymetry\dataset\collected data\Sentinel-2\2020-10-08-00_00_2020-10-08-23_59_AbuDhabi.tiff")
rds = rds[0]
raster_lats = rds.y.to_numpy()
raster_lons = rds.x.to_numpy()

shp_path = r"C:\Users\ku500817\Desktop\wassim model colab\khalifa\Result-Obs-Mod.xlsx"
shp = pd.read_csv(shp_path)  # EPSG


fig, ax = plt.subplots()
# img = rds.to_numpy()
# img = np.transpose(img, (1, 2, 0))
# rds.plot.imshow(cmap="Greys",ax=ax)
#ax.set_title("Visualize shp points and pixels")
#ax.set_axis_off()

ax.plot(shp['Longitude'],  shp['Latitude'], 'r*')

plt.show()