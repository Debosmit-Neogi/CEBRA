import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from cebra import CEBRA
import torch

from matplotlib.collections import LineCollection
import pandas as pd
import cebra.integrations.plotly
from numpy import genfromtxt

cebra.allow_lazy_imports()
hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')

# cal_data = hippocampus_pos.neural.numpy()[:]

# print(cal_data.shape)

# behav_data = hippocampus_pos.continuous_index[:,:]

# print(behav_data.shape)

calc_file = 'data/fear_training_calcium.csv'
cal_data = genfromtxt(calc_file, delimiter=',')

cal_data = cal_data[:,1:]
print(cal_data.shape)

behav_file = 'data/resampled_fear_train.csv'

behav_data = genfromtxt(behav_file, delimiter=',')
behav_data = behav_data[:6144, 1:]

behav_data = torch.tensor(behav_data, dtype=torch.float32)
print(behav_data.shape)

# fig = plt.figure(figsize=(9,3), dpi=150)
# plt.subplots_adjust(wspace = 0.3)
# ax = plt.subplot(121)
# ax.imshow(cal_data[:1000].T, aspect = 'auto', cmap = 'gray_r')
# plt.ylabel('Neuron #')
# plt.xlabel('Time [s]')
# plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))

# ax2 = plt.subplot(122)
# ax2.scatter(np.arange(1000), behav_data[:1000,0], c = 'gray', s=1)
# plt.ylabel('Position [m]')
# plt.xlabel('Time [s]')
# plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))
# plt.show()

max_iterations = 10000
output_dimension = 32

cebra_posdir3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='euclid',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_posdir3_model.fit(cal_data,behav_data)
cebra_posdir3 = cebra_posdir3_model.transform(cal_data)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Assuming `cebra_posdir3` is your 3D embedding and `behav_data[:, 0]` are the labels
# embedding = cebra_posdir3  # Replace with your actual embedding variable
# labels = behav_data[:, 0]  # Replace with your actual labels
# labels1 = behav_data[:, 1]

# # Normalize the labels for colormap mapping
# # norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
# # norm1 = plt.Normalize(vmin=labels1.min(), vmax=labels1.max())
# colors = labels
# colors1 = labels1

# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, s=50, alpha=0.7)
# sc1 = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors1, s=50, alpha=0.7)

# # Add color bar
# cbar = plt.colorbar(cm.ScalarMappable(labels, cmap='rainbow'), ax=ax)
# cbar.set_label('Behavior 1')
# cbar1 = plt.colorbar(cm.ScalarMappable(labels1, cmap='plasma'), ax=ax)
# cbar1.set_label('Behavior 2')

# # Add title and labels
# ax.set_title('CEBRA-Behavior Embedding')
# ax.set_xlabel('Embedding Dimension 1')
# ax.set_ylabel('Embedding Dimension 2')
# ax.set_zlabel('Embedding Dimension 3')

# # Show the plot
# plt.show()
embedding = cebra_posdir3  # Replace with your actual embedding variable
labels = behav_data[:, 0]  # First behavior label
labels1 = behav_data[:, 1]  # Second behavior label

# Create a 3D scatter plot
fig = plt.figure(figsize=(15, 7))

# First subplot for Behavior 1
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap='rainbow', s=50, alpha=0.7)
cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.5, aspect=5)
cbar1.set_label('Behavior 1')
ax1.set_title('3D Scatter Plot for Behavior 1')
ax1.set_xlabel('Embedding Dimension 1')
ax1.set_ylabel('Embedding Dimension 2')
ax1.set_zlabel('Embedding Dimension 3')

# Second subplot for Behavior 2
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels1, cmap='plasma', s=50, alpha=0.7)
cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.5, aspect=5)
cbar2.set_label('Behavior 2')
ax2.set_title('3D Scatter Plot for Behavior 2')
ax2.set_xlabel('Embedding Dimension 1')
ax2.set_ylabel('Embedding Dimension 2')
ax2.set_zlabel('Embedding Dimension 3')

plt.tight_layout()
plt.show()