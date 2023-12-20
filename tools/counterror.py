import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

shp_path = r"C:\Users\ku500817\Desktop\bathymetry\code\TransBathy_results\trainingcurvers\test_visual_output_sanjuan_nov6.csv"
out_path = r"C:\Users\ku500817\Desktop\bathymetry\code\TransBathy_results\trainingcurvers\sanjuan_bias_nov6.xlsx"

shp = pd.read_csv(shp_path)
shp_prd = shp['predictions'].to_numpy()
shp_depth = shp['observation'].to_numpy()
shp_diff = abs(shp_depth - shp_prd)

max_depth = np.min(shp_depth)
n_loops = abs(max_depth) // 5 + 1

count = []
x = []
y = []
e = []
upbound = -5
for i in range(int(n_loops)):
    upbound = i * -5
    lowbound = (i + 1) * -5
    if i == 0:
        arr = shp_diff[np.where(shp_depth >= lowbound)]
    elif i == n_loops - 1:
        arr = shp_diff[np.where(shp_depth < upbound)]
    else:
        arr = shp_diff[np.where(np.logical_and(shp_depth >= lowbound, shp_depth < upbound))]

    if not arr.size == 0:
        count.append(arr)
        x.append(lowbound)
        y.append(np.mean(arr))
        e.append(np.std(arr))


print(x)
print(y)
print(e)
dict = {}
dict['depth'] = x
dict['mean'] = y
dict['error'] = e
df = pd.DataFrame(dict)
df.to_excel(out_path)
# plt.errorbar(x, y, e, linestyle='None', marker='^')
# plt.show()
print('done')



# count_5 = shp_diff[np.where(shp_depth >= -5)]
# count_10 = shp_diff[np.where(np.logical_and(shp_depth>=-10, shp_depth<-5))]
# count_15 = shp_diff[np.where(np.logical_and(shp_depth>=-15, shp_depth<-10))]
# count_20 = shp_diff[np.where(np.logical_and(shp_depth>=-20, shp_depth<-15))]
# count_25 = shp_diff[np.where(np.logical_and(shp_depth>=-25, shp_depth<-20))]
# count_30 = shp_diff[np.where(np.logical_and(shp_depth>=-30, shp_depth<-25))]
# count_35 = shp_diff[np.where(np.logical_and(shp_depth>=-35, shp_depth<-30))]
# count_40 = shp_diff[np.where(np.logical_and(shp_depth>=-40, shp_depth<-35))]
# count_45 = shp_diff[np.where(np.logical_and(shp_depth>=-45, shp_depth<-40))]
# #count_50 = shp_diff[np.where(shp_depth<-45)]

# x = [5,10,15,20,25,30,35,40,45]
# y = [np.mean(count_5),np.mean(count_10),np.mean(count_15),np.mean(count_20),np.mean(count_25),
#      np.mean(count_30),np.mean(count_35),np.mean(count_40),np.mean(count_45)]
# e = [np.std(count_5),np.std(count_10),np.std(count_15),np.std(count_20),np.std(count_25),
#      np.std(count_30),np.std(count_35),np.std(count_40),np.std(count_45)]