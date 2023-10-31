import xarray as xr
import pandas as pd
import numpy as np

arr = xr.DataArray(np.random.rand(4, 3), [('time', pd.date_range('2000-01-01', periods=4)), ('space', ['IA', 'IL', 'IN'])])

print('Done')