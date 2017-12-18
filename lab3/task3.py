# Author: Michail Rudanov <https://github.com/mrudanov>

from sklearn.metrics import adjusted_rand_score


def get_adjusted_rand_index(labels_true, labels_predicted):
    return adjusted_rand_score(labels_true, labels_predicted)


def task3(labels_true, labels_predicted):
    return get_adjusted_rand_index(labels_true, labels_predicted)
