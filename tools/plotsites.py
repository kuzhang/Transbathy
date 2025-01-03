from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

m = Basemap(width=12000000,height=9000000, projection='lcc',
            resolution=None, lat_1=45, lat_2= 55, lat_0= 24.3, lon_0=53.5)
m.bluemarble()
plt.show()
#25.34 52.09  23.83 55.50
#urcrnrlat=25.02,llcrnrlat=23.87,urcrnrlon=54.5,llcrnrlon=52.09,