# Author: Evgeny Kharitonov <https://github.com/mors741>
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from collections import defaultdict
from matplotlib.ticker import AutoMinorLocator


MARKERS = {-1: ".", 0: ",", 1: "o", 2: "v", 3: "^", 4: "<",
           5: ">", 6: "*", 7: "x", 8: "+", 9: "4", 10: "_", 11: "s", 12: "p", 13: "P", 14: "1", 15: "h", 16: "H",
           17: "D", 18: "2", 19: "X", 20: "3", 21: "d", 22: "|", 23: "8"}


def count_clusters_and_noise(labels):
    clusters = set()
    noise = 0
    for label in labels:
        if label == -1:
            noise += 1
        else:
            clusters.add(label)
    return len(clusters), noise


def dbscan_params(eps, m_s, data):
    db = DBSCAN(eps=eps, min_samples=m_s).fit(data)
    k, noise = count_clusters_and_noise(db.labels_)
    if k > 1:
        chi = calinski_harabaz_score(data, db.labels_)
        # si = silhouette_score(data, db.labels_)
        si = 0
    else:
        chi = 1
        si = -1
    return k, noise, chi, si


def dbscan_visualize(eps, m_s, data, title):
    db = DBSCAN(eps=eps, min_samples=m_s).fit(data)
    visualize(data, db.labels_, title)


def cluster_coordinates(data, labels):
    x_clusters = defaultdict(list)
    y_clusters = defaultdict(list)
    for i in range(len(data)):
        x_clusters[labels[i]].append(data[i][0])
        y_clusters[labels[i]].append(data[i][1])
    return x_clusters, y_clusters


def visualize(data, labels, title="Visualization"):
    plt.figure()
    x_clusters, y_clusters = cluster_coordinates(data, labels)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in range(len(x_clusters)):
        label = list(x_clusters)[i]
        if label >= 0:
            desc = "Cluster " + str(label)
        else:
            desc = "Noise"
        plt.scatter(x_clusters[label], y_clusters[label], label=desc)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()


def print_max(name, max_val, E, M, K, N, CHI, SI):
    i, j = max_val[1:]
    print (name + ": [eps=" + str(E[i][j]) + ", m_s=" + str(M[i][j]) + ", K=" + str(K[i][j]) \
          + ", noise=" + str(N[i][j]) + ", CHI=" + str(CHI[i][j]) + ", SI=" + str(SI[i][j]) + "]")


def task2(data, eps_range, ms_range):
    E, M = np.meshgrid(eps_range, ms_range)
    shape = np.shape(E)
    K = np.empty(shape)
    N = np.empty(shape)
    CHI = np.empty(shape)
    SI = np.empty(shape)
    max_chi = (0, -1, -1)  # (chi, eps, m_s)
    max_si = (-1, -1, -1)  # (si, eps, m_s)
    for i in range(shape[0]):
        print (str(M[i][0]) + "/" + str(M[-1][0]))
        for j in range(shape[1]):
            print(str(E[i][j]) + " : " + str(M[i][j]))
            k, noise, chi, si = dbscan_params(E[i][j], M[i][j], data)
            K[i][j] = k
            N[i][j] = noise
            CHI[i][j] = chi
            SI[i][j] = si
            if chi > max_chi[0]:
                max_chi = (chi, i, j)
            if si > max_si[0]:
                max_si = (si, i, j)

    print_max("max_chi", max_chi, E, M, K, N, CHI, SI)
    print_max("max_si", max_si, E, M, K, N, CHI, SI)

    if np.shape(data)[1] == 2:
        dbscan_visualize(E[max_chi[1]][max_chi[2]], M[max_chi[1]][max_chi[2]], data,
                         "Max CHI (" + "{:.1f}".format(CHI[max_chi[1]][max_chi[2]]) + ")")
        dbscan_visualize(E[max_si[1]][max_si[2]], M[max_si[1]][max_si[2]], data,
                         "Max SI (" + "{:.3f}".format(SI[max_si[1]][max_si[2]]) + ")")
    else:
        print("Couldn't visualize clusterization result. Dimensions != 2")
