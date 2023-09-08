# import matplotlib.pyplot as plt
# import rioxarray
# import numpy as np
#
# rds = rioxarray.open_rasterio(r"C:\Users\ku500817\Desktop\bathymetry\dataset\From Fahim\insitu\clip\dalma_clip\dalma_clip.tif")
#
# print("The CRS for this data is:", rds.rio.crs)
# print("The spatial extent is:", rds.rio.bounds())
# print("The no data value is:", rds.rio.nodata)
# print("the minimum raster value is: ", np.nanmin(rds.values))
# print("the maximum raster value is: ", np.nanmax(rds.values))
# print('dimension{}'.format(rds.shape))
# # rds.plot()
# # plt.show()
#
# fig, ax = plt.subplots(figsize=(10, 5))
# rds[1,:,:].plot(ax = ax)
# ax.set_title("Lidar Digital Elevation Model (DEM) \n Boulder Flood 2013")
# ax.set_axis_off()
# plt.show()
#
# rds.plot.hist(color="purple")
# plt.show()
#
# rds = rds.squeeze().drop("spatial_ref").drop("band")
# rds.name = "data"
# res = rds.to_dataframe().reset_index()
# res.head(3)
#
# gr = res.groupby(res.band)
# gr.get_group('0').head()
#
# fig, ax = plt.subplots(figsize=(10, 5))
# show(dataset.read(5), transform=dataset.transform,ax=ax)
# ax.plot(res.x,res.y,'ro', markersize=3)
#
# print('done')
#
#
