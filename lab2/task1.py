import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from lab2.kernel_functions import box_kernel


def interquartile_range(down_quartile_val, up_quartile_val):
    return up_quartile_val - down_quartile_val


def down_quartile(sorted_data):
    l = len(sorted_data)
    down = sorted_data[int((2 + l) / 4)]
    up = sorted_data[np.math.ceil((2 + l) / 4)]
    return (down + up) / 2


def up_quartile(sorted_data):
    l = len(sorted_data)
    down = sorted_data[int((2 + 3 * l) / 4)]
    up = sorted_data[np.math.ceil((2 + 3 * l) / 4)]
    return (down + up) / 2


def silverman_bandwidth(data):
    sorted_data = sorted(data)
    return 0.9 * min(np.std(data), interquartile_range(down_quartile(sorted_data), up_quartile(sorted_data)) / 1.34) * np.math.pow(len(data), -0.2)


def general_naive_density_estimator(x, data, bandwidth, kernel_function):
    n = len(data)
    kernel_sum = 0
    for i in range(n):
        kernel_sum += kernel_function((x - data[i]) / bandwidth)
    return 1 / (n * bandwidth) * kernel_sum


def task1(data):
    sorted_data = sorted(data)
    silverman_band = silverman_bandwidth(data)
    print(silverman_band)
    small_band = 0.1
    large_band = 3
    val = 0.

    ins = []
    silverman_outs = []
    small_outs = []
    large_outs = []
    for x in sorted_data:
        ins.append(x)
        silverman_outs.append(general_naive_density_estimator(x, data, silverman_band, box_kernel))
        small_outs.append(general_naive_density_estimator(x, data, small_band, box_kernel))
        large_outs.append(general_naive_density_estimator(x, data, large_band, box_kernel))

    plt.figure(2)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(ins, small_outs)
    plt.plot(sorted_data, np.zeros_like(sorted_data) + val, 'x', color='orange')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    plt.figure(3)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(ins, silverman_outs)
    plt.plot(sorted_data, np.zeros_like(sorted_data) + val, 'x', color='orange')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    plt.figure(4)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(ins, large_outs)
    plt.plot(sorted_data, np.zeros_like(sorted_data) + val, 'x', color='orange')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
