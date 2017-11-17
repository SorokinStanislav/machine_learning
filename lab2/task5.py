from lab2.task4 import variances, biases
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def task5(data):
    all_box_variances = []
    all_gauss_variances = []
    all_epanechnikov_variances = []
    all_triangular_variances = []
    all_box_biases = []
    all_gauss_biases = []
    all_epanechnikov_biases = []
    all_triangular_biases = []

    for x in np.arange(-3, 10, 0.1):
        print(str(x) + '\n')
        box_variances, gauss_variances, epanechnikov_variances, triangular_variances = variances(data, x)
        box_biases, gauss_biases, epanechnikov_biases, triangular_biases = biases(data, x)
        all_box_variances.append(box_variances)
        all_box_biases.append(box_biases)
        all_gauss_variances.append(gauss_variances)
        all_gauss_biases.append(gauss_biases)
        all_epanechnikov_variances.append(epanechnikov_variances)
        all_epanechnikov_biases.append(epanechnikov_biases)
        all_triangular_variances.append(triangular_variances)
        all_triangular_biases.append(triangular_biases)

    mise_box = []
    mise_gauss = []
    mise_epanechnikov = []
    mise_triangular = []

    for h in range(0, 99):
        print(str(h) + '\n')
        box_band_variances = []
        gauss_band_variances = []
        epanechnikov_band_variances = []
        triangular_band_variances = []
        box_band_biases = []
        gauss_band_biases = []
        epanechnikov_band_biases = []
        triangular_band_biases = []
        for x in range(0, 130):
            box_band_variances.append(all_box_variances[x][h])
            gauss_band_variances.append(all_gauss_variances[x][h])
            epanechnikov_band_variances.append(all_epanechnikov_variances[x][h])
            triangular_band_variances.append(all_triangular_variances[x][h])
            box_band_biases.append(all_box_biases[x][h] ** 2)
            gauss_band_biases.append(all_gauss_biases[x][h] ** 2)
            epanechnikov_band_biases.append(all_epanechnikov_biases[x][h] ** 2)
            triangular_band_biases.append(all_triangular_biases[x][h] ** 2)

        mise_box.append(np.trapz(box_band_variances, None, 0.1) + np.trapz(box_band_biases, None, 0.1))
        mise_gauss.append(np.trapz(gauss_band_variances, None, 0.1) + np.trapz(gauss_band_biases, None, 0.1))
        mise_epanechnikov.append(np.trapz(epanechnikov_band_variances, None, 0.1) + np.trapz(epanechnikov_band_biases, None, 0.1))
        mise_triangular.append(np.trapz(triangular_band_variances, None, 0.1) + np.trapz(triangular_band_biases, None, 0.1))

    plt.figure(29)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), mise_box, label='Прямоугольное окно')
    plt.plot(np.arange(0.05, 5, 0.05), mise_gauss, label='Гауссово окна')
    plt.plot(np.arange(0.05, 5, 0.05), mise_epanechnikov, label='Окно Епанечникова')
    plt.plot(np.arange(0.05, 5, 0.05), mise_triangular, label='Треугольное окно')
    plt.xlabel('h')
    plt.ylabel('MISE(h)')
    plt.legend()
    plt.show()
