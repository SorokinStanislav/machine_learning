# Author: Michail Rudanov <https://github.com/mrudanov>

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# data - np.array
# data should have structure: feature 1, feature 2, ..., feature N, cluster_number
def task4(data, distance='euclidean'):
    # Sort by cluster number column
    cluster_index = data.shape[1] - 1
    ind = np.argsort(data[:, cluster_index])
    data = data[ind]
    data = np.delete(data, cluster_index, axis=1)

    # calculate distances matrix
    if distance == 'euclidean':
        dist = euclidean_distances(data, data)
        plot_matrix_heatmap(dist)
    if distance == 'manhattan':
        dist = manhattan_distances(data, data)
        plot_matrix_heatmap(dist)


def plot_matrix_heatmap(data, title='Cluster heatmap', cmap=plt.get_cmap('hot')):
    fig, ax = plt.subplots()
    plt.imshow(data, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.ylabel('Номер примера', fontsize=12)
    plt.xlabel('Номер примера', fontsize=12)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.show()




