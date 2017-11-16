import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from lab2.kernel_functions import box_kernel
from lab2.variance import variance
from lab2.task1 import silverman_bandwidth, general_naive_density_estimator
from lab2.variance import random_half_data


def task3(data):
    small_band = 0.1
    large_band = 3
    ins = []
    silverman_variances = []
    small_variances = []
    large_variances = []

    for x in range(1, 50):
        ins.append(x)
        silverman_outs = []
        small_outs = []
        large_outs = []
        rand_data = sorted(random_half_data(data))
        silverman_band = silverman_bandwidth(rand_data)
        for rand_x in rand_data:
            silverman_outs.append(general_naive_density_estimator(rand_x, rand_data, silverman_band, box_kernel))
            small_outs.append(general_naive_density_estimator(rand_x, rand_data, small_band, box_kernel))
            large_outs.append(general_naive_density_estimator(rand_x, rand_data, large_band, box_kernel))
        silverman_variances.append(variance(silverman_outs))
        small_variances.append(variance(small_outs))
        large_variances.append(variance(large_outs))

    plt.figure(9)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(ins, small_variances)
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
    plt.plot(ins, silverman_variances)
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
    plt.plot(ins, large_variances)
    plt.xlabel('x')
    plt.ylabel('Variance(x)')
    plt.show()
