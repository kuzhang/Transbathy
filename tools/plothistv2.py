import yaml
import numpy as np
from scipy import stats
import pandas as pd
import os
import rioxarray as rxr
import json
import matplotlib.pyplot as plt

shp_path = r"C:\Users\ku500817\Desktop\bathymetry\code\TransBathy_results\trainingcurvers\test_rmse_lidarnabu_epoch20_span2_nov6.csv"
shp = pd.read_csv(shp_path)
shp_data = shp['Depth'].to_numpy()


plt.hist(shp_data, bins=100)
plt.xlabel('depth')
plt.show()