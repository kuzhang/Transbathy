import numpy as np
import pandas as pd

file_path = r"C:\Users\ku500817\Desktop\bathymetry\code\TransBathy_results\test_visual_guam_output.csv"
df = pd.read_csv(file_path)

obs = df['observation']
prd = df['predictions']

corr_matrix = np.corrcoef(obs, prd)
corr = corr_matrix[0,1]
R_sq = corr**2
print('R square:{}'.format(R_sq))