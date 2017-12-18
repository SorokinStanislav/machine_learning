from numpy import concatenate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import calinski_harabaz_score, silhouette_score


def k_means(data, n_clusters):
    cluster_result = KMeans(n_clusters=n_clusters, init='k-means++', tol=1e-2).fit(data)
    labels = cluster_result.labels_
    centers = cluster_result.cluster_centers_
    interia = cluster_result.inertia_
    return labels, centers, interia


def show_cluster_scatter_plot(data, labels, n_clusters):
    data_with_labels = concatenate((data, labels[:, None]), axis=1)
    plt.figure(1)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in range(0, n_clusters):
        x = []
        y = []
        for point in data_with_labels:
            if int(point[2]) == i:
                x.append(point[0])
                y.append(point[1])

        plt.scatter(x, y, label="Кластер " + str(i + 1))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def calinski_harabaz_index(data, labels):
    return calinski_harabaz_score(data, labels)


def silhouette_index(data, labels):
    return silhouette_score(data, labels, metric='euclidean')


def task1(data, max_clusters):
    interias = []
    calinski_indexes = []
    silhouette_indexes = []

    max_chi = 0
    max_si = 0
    k_1 = 0
    k_2 = 0

    for n_cluster in range(2, max_clusters + 1):
        print(n_cluster)
        labels, centers, interia = k_means(data, n_cluster)
        interias.append(interia)

        chi = calinski_harabaz_index(data, labels)
        si = silhouette_index(data, labels)
        calinski_indexes.append(chi)
        silhouette_indexes.append(si)

        if chi > max_chi:
            max_chi = chi
            k_1 = n_cluster

        if si > max_si:
            max_si = si
            k_2 = n_cluster

        if data.shape[1] == 2:
            show_cluster_scatter_plot(data, labels, n_cluster)

    plt.figure(2)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('S(K)')
    plt.plot(range(2, max_clusters + 1), interias)
    plt.xlabel("K", fontsize=14)
    plt.ylabel("S", fontsize=14)
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(3)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('CHI(K)')
    plt.plot(range(2, max_clusters + 1), calinski_indexes)
    plt.xlabel("K", fontsize=14)
    plt.ylabel("CHI", fontsize=14)
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(4)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('SI(K)')
    plt.plot(range(2, max_clusters + 1), silhouette_indexes)
    plt.xlabel("K", fontsize=14)
    plt.ylabel("SI", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


