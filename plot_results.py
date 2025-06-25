import pickle
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_path='prediction_results/output_KLa_extrainputs'
images_base_folder = 'images_pileaute_kla'

from os import listdir
from os.path import isdir, join

image_folders = sorted([
    folder for folder in listdir(images_base_folder)
    if isdir(join(images_base_folder, folder)) and not folder.startswith('.')
])

all_labels = []
all_preds = []
for output in listdir(output_path):
    if output.endswith('.pk'):
        with open(os.path.join(output_path, output), 'rb') as f:
            data = pickle.load(f)
            labels=data[5]
            preds=data[6]
            all_labels.append(labels)
            all_preds.append(preds)

#calc average per fold
avg_preds = np.mean([all_preds[i] for i in range(4)], axis=0)
avg_labels= np.mean([all_labels[i] for i in range(4)], axis=0)

y_predicted = []
y_true = []
std_dev = []
i=0
for folder in image_folders:
    path = f"{images_base_folder}/{folder}/basin5/10x"
    images_list = listdir(path)
    temporary_pred=[]
    temporary_label=[]
    for image in images_list:
        pred = avg_preds[i]
        label=avg_labels[i]
        i+=1
        temporary_pred.append(pred)
        temporary_label.append(label)
    y_predicted.append(sum(temporary_pred)/len(temporary_pred))
    y_true.append(sum(temporary_label)/len(temporary_label))
    std_dev.append(np.std(temporary_pred))

std_dev = np.array(std_dev)
y_predicted = np.array(y_predicted)

y_pred_upper = y_predicted.flatten() + std_dev
y_pred_lower = y_predicted.flatten() - std_dev

#y_true=pd.read_csv('data/SVI/cleaned_SVIs_interpolated.csv', index_col=0)
y_true = pd.Series(y_true, index=pd.to_datetime(image_folders))
y_predicted = pd.DataFrame(y_predicted, index=pd.to_datetime(image_folders))


#### make plot
# Define train and test indices
train_indices_part1 = list(range(0, 64))      
#train_indices_part2 = list(range(64, 87))      
test_indices = list(range(64, 87))            

plt.figure(figsize=(14, 3), dpi=150)
plt.plot(y_true, '.-', label='Measurements', color='blue')

# Plot model predictions
plt.plot(y_predicted.iloc[train_indices_part1], '.-', label='Model predictions (train)', color='orange')
#plt.plot(y_predicted.iloc[train_indices_part2], '.-', color='orange')
plt.plot(y_predicted.iloc[test_indices], '.-', label='Model predictions (test)', color='red')

# Plot Std Dev Band for Train – Part 1
plt.fill_between(y_predicted.index[train_indices_part1],
                 y_pred_lower[train_indices_part1],
                 y_pred_upper[train_indices_part1],
                 color='orange', alpha=0.2, zorder=1)

# Plot Std Dev Band for Train – Part 2
#plt.fill_between(y_predicted.index[train_indices_part2],
                 #y_pred_lower[train_indices_part2],
                 #y_pred_upper[train_indices_part2],
                 #color='orange', alpha=0.2, zorder=1)

# Plot Std Dev Band for Test
plt.fill_between(y_predicted.index[test_indices],
                 y_pred_lower[test_indices],
                 y_pred_upper[test_indices],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("Qair")
plt.title("Qair residuals (CNN convnext_nano)")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error

# Zet y_true en y_predicted om naar NumPy arrays
y_true_array = y_true.to_numpy()
y_pred_array = y_predicted.to_numpy().flatten()

# Bereken MSE's
# Combineer de twee trainingsets
train_indices = train_indices_part1 #+ train_indices_part2

# Bereken MSE voor gecombineerde training
mse_train = mean_squared_error(y_true_array[train_indices], y_pred_array[train_indices])
mse_test  = mean_squared_error(y_true_array[test_indices], y_pred_array[test_indices])
mse_total = mean_squared_error(y_true_array, y_pred_array)

# Print de resultaten
print(f"Train MSE:  {mse_train:.4f}")
print(f"Test MSE:   {mse_test:.4f}")
print(f"Totaal MSE: {mse_total:.4f}")

# plot_split_index=64 #nr of training folders
# plt.figure(figsize=(14, 3), dpi=150)
# plt.plot(y_true, '.-', label='Measurements', color='blue')
# plt.plot(y_predicted.iloc[:plot_split_index], '.-', label='Model predictions (train)', color='orange')
# plt.plot(y_predicted.iloc[plot_split_index:], '.-', label='Model predictions (test)', color='red')

# # Plot Standard Deviation Band (Train) - use iloc
# plt.fill_between(y_predicted.index[:plot_split_index],
#                  y_pred_lower[:plot_split_index],
#                  y_pred_upper[:plot_split_index],
#                  color='orange', alpha=0.2, zorder=1)

# # Plot Standard Deviation Band (Test) - use iloc
# plt.fill_between(y_predicted.index[plot_split_index:],
#                  y_pred_lower[plot_split_index:],
#                  y_pred_upper[plot_split_index:],
#                  color='red', alpha=0.2, zorder=1)

# plt.xlabel("Time")
# plt.ylabel("Qair")
# plt.title("Qair residuals (CNN convnext_nano)")
# plt.legend()
# plt.show()