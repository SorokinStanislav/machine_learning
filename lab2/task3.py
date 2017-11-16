import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from lab2.kernel_functions import box_kernel
from lab2.variance import variance
from lab2.bias import bias, true_pdf
from lab2.task1 import silverman_bandwidth, general_naive_density_estimator
from lab2.variance import random_half_data


def task3(data):
    small_band = 0.1
    large_band = 3
    silverman_densities = np.zeros((130, 50))
    small_densities = np.zeros((130, 50))
    large_densities = np.zeros((130, 50))
    general_counter = 0
    counter = 0

    for k in range(0, 50):
        general_counter += counter
        counter = 0
        for x in np.arange(-3, 10, 0.1):
            rand_data = sorted(random_half_data(data))
            silverman_band = silverman_bandwidth(rand_data)
            silverman_densities[counter][k] = general_naive_density_estimator(x, rand_data, silverman_band, box_kernel)
            small_densities[counter][k] = general_naive_density_estimator(x, rand_data, small_band, box_kernel)
            large_densities[counter][k] = general_naive_density_estimator(x, rand_data, large_band, box_kernel)
            counter += 1

    silverman_variances = []
    small_variances = []
    large_variances = []
    for k in range(0, 130):
        silverman_variances.append(variance(silverman_densities[k]))
        small_variances.append(variance(small_densities[k]))
        large_variances.append(variance(large_densities[k]))

    plt.figure(9)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(-3, 10, 0.1), small_variances)
    plt.xlabel('x')
    plt.ylabel('Variance(x)')
    plt.show()

    plt.figure(10)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(-3, 10, 0.1), silverman_variances)
    plt.xlabel('x')
    plt.ylabel('Variance(x)')
    plt.show()

    plt.figure(11)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(-3, 10, 0.1), large_variances)
    plt.xlabel('x')
    plt.ylabel('Variance(x)')
    plt.show()

    silverman_biases = []
    small_biases = []
    large_biases = []
    silverman_band = silverman_bandwidth(data)
    for x in np.arange(-3, 10, 0.1):
        silverman_biases.append(abs(true_pdf(x) - general_naive_density_estimator(x, data, silverman_band, box_kernel)))
        small_biases.append(abs(true_pdf(x) - general_naive_density_estimator(x, data, small_band, box_kernel)))
        large_biases.append(abs(true_pdf(x) - general_naive_density_estimator(x, data, large_band, box_kernel)))

    plt.figure(12)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(-3, 10, 0.1), small_biases)
    plt.xlabel('x')
    plt.ylabel('Absolute Bias(x)')
    plt.show()

    plt.figure(13)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(-3, 10, 0.1), silverman_biases)
    plt.xlabel('x')
    plt.ylabel('Absolute Bias(x)')
    plt.show()

    plt.figure(14)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(-3, 10, 0.1), large_biases)
    plt.xlabel('x')
    plt.ylabel('Absolute Bias(x)')
    plt.show()
