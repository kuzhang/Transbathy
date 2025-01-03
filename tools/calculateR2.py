import numpy as np
import pandas as pd

file_path = r"C:\Users\ku500817\Desktop\research\bathymetry\code\TransBathy_results\trainingcurvers\global\test_outputs_lidarnabu_epoch20_span2_nov21.csv"
df = pd.read_csv(file_path)

obs = df['observation']
prd = df['predictions']

corr_matrix = np.corrcoef(obs, prd)
corr = corr_matrix[0,1]
R_sq = corr**2
print('R square:{}'.format(R_sq))

# Calculate RMSE
rmse = np.sqrt(np.mean((obs - prd) ** 2))

print(f"RMSE: {rmse}")