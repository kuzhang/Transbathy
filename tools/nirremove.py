import rioxarray as rxr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import geopandas as gpd
import os

rds = rxr.open_rasterio(r"C:\Users\ku500817\Desktop\bathymetry\dataset\From Fahim\insitu\abu2\abudhabi\2019-12-28-00_00_2019-12-28-23_59_Sentinel-2_L2A_B8A_(Raw).tiff")
rds = rds.rio.reproject("EPSG:4326")
img = rds.to_numpy().squeeze(0).astype(np.float32)
img[img==rds.rio.nodata] = np.nan
mask_arr = np.ma.masked_less(img, 250)
mask = mask_arr.mask

raster_lats = rds.y.to_numpy()
raster_lons = rds.x.to_numpy()
x, y = np.meshgrid(raster_lons, raster_lats)

#shp_path = r"C:\Users\ku500817\Desktop\bathymetry\dataset\From Fahim\insitu\abu2\abu2.shp"
shp_path = r"C:\Users\ku500817\Desktop\bathymetry\dataset\TrainingData\clip\abudhabi\abudhabi.csv"
#geodf = gpd.read_file(shp_path)
shp = pd.read_csv(shp_path)
#clipped = rds.rio.clip(geodf.geometry.values, geodf.crs)

# raster_cords = np.array([x.ravel(), y.ravel()]).transpose()
# shp_cords = np.array([shp.Longitude.values, shp.Latitude.values]).transpose()
shp_depth = shp.Depth

# lon = raster_lons[0]
# err = 0
# for i in range(1, len(raster_lons)):
#     err += raster_lons[i] - lon
#     lon = raster_lons[i]
# print('lon gaps{}'.format(err/len(raster_lons)))
#
# lat = raster_lats[0]
# err = 0
# for i in range(1, len(raster_lats)):
#     err += raster_lats[i] - lat
#     lat = raster_lats[i]
# print('lon gaps{}'.format(err/len(raster_lats)))

drop_list = []
for i, (x1, y1) in enumerate(zip(shp.Longitude.values, shp.Latitude.values)):
    lon_idx = np.argmin(np.abs(raster_lons - x1))
    lat_idx = np.argmin(np.abs(raster_lats - y1))
    #print('data{},{}'.format(raster_lats[lat_idx], raster_lons[lon_idx]))

    if not mask[lat_idx, lon_idx]:
        drop_list.append(i)

shp.drop(shp.index[drop_list], axis='index')


dict = {
    'Longitude': x.ravel(),
    'Latitude': y.ravel(),
    'Mask': mask.ravel()
}

df = pd.DataFrame(dict)
output_path = r"C:\Users\ku500817\Desktop\bathymetry\dataset\From Fahim\insitu\abu2\mask.csv"
df.to_csv(output_path)



fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img,cmap='gray',vmin=0, vmax=255)
fig.add_subplot(1,2,2)
plt.imshow(mask, cmap='gray',vmin=0, vmax=1)
plt.show()

print('Done')
