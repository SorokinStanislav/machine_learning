import numpy as np
import matplotlib.pyplot as plt
import lab2.kernel_functions as kernel
from matplotlib.ticker import AutoMinorLocator

from lab2.task1 import silverman_bandwidth, general_naive_density_estimator


def task2(data):
    sorted_data = sorted(data)
    bandwidth = silverman_bandwidth(data)
    val = 0.

    ins = []
    box_outs = []
    gauss_outs = []
    epanechnikov_outs = []
    triangular_outs = []
    for x in sorted_data:
        ins.append(x)
        box_outs.append(general_naive_density_estimator(x, data, bandwidth, kernel.box_kernel))
        gauss_outs.append(general_naive_density_estimator(x, data, bandwidth, kernel.gaussian_kernel))
        epanechnikov_outs.append(general_naive_density_estimator(x, data, bandwidth, kernel.epanechnikov_kernel))
        triangular_outs.append(general_naive_density_estimator(x, data, bandwidth, kernel.triangular_kernel))

    plt.figure(5)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(ins, box_outs)
    plt.plot(sorted_data, np.zeros_like(sorted_data) + val, 'x', color='orange')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    plt.figure(6)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(ins, gauss_outs)
    plt.plot(sorted_data, np.zeros_like(sorted_data) + val, 'x', color='orange')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    plt.figure(7)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(ins, epanechnikov_outs)
    plt.plot(sorted_data, np.zeros_like(sorted_data) + val, 'x', color='orange')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    plt.figure(8)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(ins, triangular_outs)
    plt.plot(sorted_data, np.zeros_like(sorted_data) + val, 'x', color='orange')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
