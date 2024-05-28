from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8,5))

m = Basemap(projection='cyl',
           llcrnrlat = -90,
           urcrnrlat = 90,
           llcrnrlon = -180,
           urcrnrlon = 180,
           resolution = 'c')

m.drawcoastlines()
m.drawcountries()
# m.drawstates(color='blue')
# m.drawcounties(color='orange')
# m.drawrivers(color='blue')

# m.drawmapboundary(color='pink', linewidth=10, fill_color='aqua')
# m.fillcontinents(color='lightgreen', lake_color='aqua')

# m.drawlsmask(land_color='lightgreen', ocean_color='aqua', lakes=True)
#
# m.etopo()
# m.bluemarble()
# m.shadedrelief()

m.drawparallels(np.arange(-90,90,20),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])

# # Map (long, lat) to (x, y) for plotting
# x, y = m(-122.3, 47.6)
# plt.plot(x, y, 'or', markersize=5)
# plt.text(x, y, ' Seattle', fontsize=12);

# Map (long, lat) to (x, y) for plotting
x, y = m(54.36556268043377, 24.48927993193798)
plt.plot(x, y, 'ob', markersize=5)
plt.text(x, y, ' Abu Dhabi', fontsize=12, color='b');

# Map (long, lat) to (x, y) for plotting
x, y = m(-66.55473014545616, 18.238238640210724)
plt.plot(x, y, 'ob', markersize=5)
plt.text(x, y, 'Puerto Rico', fontsize=12, color='b');


# Map (long, lat) to (x, y) for plotting
x, y = m(-81.77413476331392, 24.560123504476245)
plt.plot(x, y, 'ob', markersize=5)
plt.text(x, y, 'Florida', fontsize=12, color='b');


# Map (long, lat) to (x, y) for plotting
x, y = m(-157.86754313306142, 21.32634452726114)
plt.plot(x, y, 'ob', markersize=5)
plt.text(x, y, 'Honolulu', fontsize=12, color='b');

# Map (long, lat) to (x, y) for plotting
x, y = m(144.74661936566582, 13.435666029046773)
plt.plot(x, y, 'ob', markersize=5)
plt.text(x, y, 'Guam', fontsize=12, color='b');


# np.arange(start,stop,step)
# labels=[left,right,top,bottom]

plt.title('Study Sites', fontsize=12)
plt.savefig('C:/Users/ku500817/Desktop/bathymetry/journal-draft/study_region.png')
plt.show()
