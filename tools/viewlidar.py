import os
import numpy as np
import pandas as pd
import rioxarray as rxr
import matplotlib.pyplot as plt

def plot_3dimg(x,y,z,idx):
    ax = plt.figure(idx).add_subplot(projection='3d')
    ax.scatter(x, y, z, marker='.')
    plt.show()

root = r"C:\Users\ku500817\Desktop\bathymetry\dataset\collected data\Lidar data\NOAA\2019_NGS_FL_topobathy_Irma_Job912253"

for path, dirs, files in os.walk(root, topdown= False):
    for name in files:
        if name.endswith('.tif'):
            img_path = os.path.join(path, name)

            with rxr.open_rasterio(img_path) as rds:
                # reproject coordinate data from utm to lon and lats
                rds = rds.rio.reproject("EPSG:4326")
                img = rds.to_numpy()

                # replace nodata (-999999.) to np.nan
                img[img==rds.rio.nodata] = np.nan

                # calculate no-nan element in raster image
                nan_num = np.count_nonzero(~np.isnan(img))
                print('Total non-NaN element is {}, covering {:.2f}% of the raster image'.format(nan_num, nan_num/img.size *100))

                img = img.squeeze()
                raster_lats = rds.y.to_numpy()
                raster_lons = rds.x.to_numpy()
                x, y = np.meshgrid(raster_lons, raster_lats)

                #view data in 3d plot
                #plot_3dimg(x, y, img_rs, 0)

                dict = {
                    'Longitude': x.ravel(),
                    'Latitude': y.ravel(),
                    'Depth': img.ravel()
                }

                df = pd.DataFrame(dict)
                # remove depth == nan row
                remove_list = np.where(np.isnan(df['Depth']))[0].tolist()
                df = df.drop(remove_list, axis='index')
                # reset index
                df.reset_index(inplace=True, drop=True)

                # save data to csv file
                output_name = name.replace('.tif','.csv')
                output_path = os.path.join(path,'csv')
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                csv_file = os.path.join(output_path,output_name)
                df.to_csv(csv_file)

print('Done')