import rioxarray
import rasterio as rio
import numpy as np

# compare rioxarray loading and rasterio loading
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

raster_img = rio.open(r"C:\Users\ku500817\Desktop\bathymetry\dataset\From Fahim\insitu\clip\dalma_clip\dalma_clip.tif")
raster_lons_uniq, raster_lats_uniq= create_raster_cord(raster_img)

print('done')