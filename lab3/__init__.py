from numpy import genfromtxt, arange, array, concatenate, append
from sklearn.cluster import DBSCAN

from lab3.task1 import task1, k_means, show_cluster_scatter_plot
from lab3.task2 import task2
from lab3.task3 import task3, get_adjusted_rand_index
from lab3.task4 import task4

FILE_NAME_1 = "resources/clust_data_8.csv"
DATA_COLUMNS_1 = [1, 2]
LABEL_COLUMN_1 = [3]
MAX_CLUSTERS_1 = 20

FILE_NAME_2 = "resources/Frogs_MFCCs.csv"
DATA_COLUMNS_2 = [i for i in range(1, 22)]
LABEL_COLUMN_2 = [25]
MAX_CLUSTERS_2 = 100


def read_data(file_name, data_columns):
    return genfromtxt(file_name, delimiter=',', skip_header=1, usecols=data_columns)


data_with_label_1 = read_data(FILE_NAME_2, DATA_COLUMNS_2 + LABEL_COLUMN_2)
data_1 = data_with_label_1[:, range(0, len(DATA_COLUMNS_2))]
labels_1 = data_with_label_1[:, data_with_label_1.shape[1] - 1]

# sum = 0
# for i in range(1, 61):
#     l = len(array(list(filter(lambda x: x[21] == i, data_with_label_1))))
#     print(str(i) + ': ' + str(l))
#     sum += l
#
# print(sum)
# for i in data_1:
#     a = 5
#     for j in i:
#         print(j)


# max_kmeans_ari = 0
# n = 0
# for n_cluster in range(4, 20):
#     kmeans_labels_1 = k_means(data_1, n_cluster)[0]
#     ari_1 = get_adjusted_rand_index(labels_1, kmeans_labels_1)
#     if ari_1 > max_kmeans_ari:
#         max_kmeans_ari = ari_1
#         n = n_cluster
# print(str(n) + ': ' + str(max_kmeans_ari))

# show_cluster_scatter_plot(data_1, labels_1, 7)

# max_dbscan_ari = 0
# best_eps = 0
# best_m_s = 0
# for eps in arange(0.005, 0.125, 0.001):
#     for m_s in range(3, 15, 1):
#         dbscan_labels_1 = DBSCAN(eps=eps, min_samples=m_s).fit(data_1).labels_
#         dbscan_ari_1 = task3(labels_1, dbscan_labels_1)
#         if dbscan_ari_1 > max_dbscan_ari:
#             max_dbscan_ari = dbscan_ari_1
#             print('new best ari: ' + str(dbscan_ari_1))
#             best_eps = eps
#             best_m_s = m_s
# print('------------------------')
# print(str(best_eps))
# print(str(best_m_s))

# kmeans_labels_1 = k_means(data_1, 20)[0]
# dbscan_labels_1 = DBSCAN(eps=0.056, min_samples=6).fit(data_1).labels_
# kmeans_ari_1 = task3(labels_1, kmeans_labels_1)
# dbscan_ari_1 = task3(labels_1, dbscan_labels_1)
# task4(data_with_label_1)
# data_for_heat_map = append(data_1, dbscan_labels_1[:, None], axis=1)
# data_for_heat_map = array(list(filter(lambda x: x[2] != -1, data_for_heat_map)))

# task4(data_for_heat_map)
# task4(data_1 + kmeans_labels_1[:, None])
# task1(data_1, MAX_CLUSTERS_2)
# task2(data_1, arange(0.001, 0.2, 0.001), range(4, 50, 1))
# task1(FILE_NAME_2, DATA_COLUMNS_2, MAX_CLUSTERS_2)
# task4(data_with_label_1, distance='manhattan')


# print(kmeans_ari_1)
# print(dbscan_ari_1)
