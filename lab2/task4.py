import numpy as np
import matplotlib.pyplot as plt
import lab2.kernel_functions as kernel
from matplotlib.ticker import AutoMinorLocator

from lab2.bias import true_pdf
from lab2.variance import variance

from lab2.task1 import general_naive_density_estimator
from lab2.variance import random_half_data


def choose_points(data):
    sorted_data = sorted(data)
    return sorted_data[33], sorted_data[50], sorted_data[212]


def variances(data, point):
    box_densities = np.zeros((99, 50))
    gauss_densities = np.zeros((99, 50))
    epanechnikov_densities = np.zeros((99, 50))
    triangular_densities = np.zeros((99, 50))

    for k in range(0, 50):
        counter = 0
        for bandwidth in np.arange(0.05, 5, 0.05):
            rand_data = sorted(random_half_data(data))
            box_densities[counter][k] = general_naive_density_estimator(point, rand_data, bandwidth, kernel.box_kernel)
            gauss_densities[counter][k] = general_naive_density_estimator(point, rand_data, bandwidth, kernel.gaussian_kernel)
            epanechnikov_densities[counter][k] = general_naive_density_estimator(point, rand_data, bandwidth, kernel.epanechnikov_kernel)
            triangular_densities[counter][k] = general_naive_density_estimator(point, rand_data, bandwidth, kernel.triangular_kernel)
            counter += 1

    box_variances = []
    gauss_variances = []
    epanechnikov_variances = []
    triangular_variances = []
    for k in range(0, 99):
        box_variances.append(variance(box_densities[k]))
        gauss_variances.append(variance(gauss_densities[k]))
        epanechnikov_variances.append(variance(epanechnikov_densities[k]))
        triangular_variances.append(variance(triangular_densities[k]))
        
    return box_variances, gauss_variances, epanechnikov_variances, triangular_variances


def biases(data, point):
    silverman_biases = []
    gauss_biases = []
    epanechnikov_biases = []
    triangular_biases = []
    for bandwidth in np.arange(0.05, 5, 0.05):
        silverman_biases.append(abs(true_pdf(point) - general_naive_density_estimator(point, data, bandwidth, kernel.box_kernel)))
        gauss_biases.append(abs(true_pdf(point) - general_naive_density_estimator(point, data, bandwidth, kernel.gaussian_kernel)))
        epanechnikov_biases.append(abs(true_pdf(point) - general_naive_density_estimator(point, data, bandwidth, kernel.epanechnikov_kernel)))
        triangular_biases.append(abs(true_pdf(point) - general_naive_density_estimator(point, data, bandwidth, kernel.triangular_kernel)))
    return silverman_biases, gauss_biases, epanechnikov_biases, triangular_biases


def task4(data):
    point_one, point_two, point_three = choose_points(data)
    box_variances_one, gauss_variances_one, epanechnikov_variances_one, triangular_variances_one = variances(data, point_one)
    box_variances_two, gauss_variances_two, epanechnikov_variances_two, triangular_variances_two = variances(data, point_two)
    box_variances_three, gauss_variances_three, epanechnikov_variances_three, triangular_variances_three = variances(data, point_three)
    
    plt.figure(15)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), box_variances_one, label='Точка 1')
    plt.plot(np.arange(0.05, 5, 0.05), box_variances_two, label='Точка 2')
    plt.plot(np.arange(0.05, 5, 0.05), box_variances_three, label='Точка 3')
    plt.xlabel('bandwidth')
    plt.ylabel('Variance(bandwidth)')
    plt.legend()
    plt.show()

    plt.figure(16)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), gauss_variances_one, label='Точка 1')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_variances_two, label='Точка 2')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_variances_three, label='Точка 3')
    plt.xlabel('bandwidth')
    plt.ylabel('Variance(bandwidth)')
    plt.legend()
    plt.show()

    plt.figure(17)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_variances_one, label='Точка 1')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_variances_two, label='Точка 2')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_variances_three, label='Точка 3')
    plt.xlabel('bandwidth')
    plt.ylabel('Variance(bandwidth)')
    plt.legend()
    plt.show()

    plt.figure(18)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), triangular_variances_one, label='Точка 1')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_variances_two, label='Точка 2')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_variances_three, label='Точка 3')
    plt.xlabel('bandwidth')
    plt.ylabel('Variance(bandwidth)')
    plt.legend()
    plt.show()

    plt.figure(19)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), box_variances_one, label='Прямоугольное окно')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_variances_one, label='Гауссово окна')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_variances_one, label='Окно Епанечникова')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_variances_one, label='Треугольное окно')
    plt.xlabel('bandwidth')
    plt.ylabel('Variance(bandwidth)')
    plt.legend()
    plt.show()
    
    plt.figure(20)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), box_variances_two, label='Прямоугольное окно')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_variances_two, label='Гауссово окна')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_variances_two, label='Окно Епанечникова')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_variances_two, label='Треугольное окно')
    plt.xlabel('bandwidth')
    plt.ylabel('Variance(bandwidth)')
    plt.legend()
    plt.show()
    
    plt.figure(21)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), box_variances_three, label='Прямоугольное окно')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_variances_three, label='Гауссово окна')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_variances_three, label='Окно Епанечникова')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_variances_three, label='Треугольное окно')
    plt.xlabel('bandwidth')
    plt.ylabel('Variance(bandwidth)')
    plt.legend()
    plt.show()

    box_biases_one, gauss_biases_one, epanechnikov_biases_one, triangular_biases_one = biases(data, point_one)
    box_biases_two, gauss_biases_two, epanechnikov_biases_two, triangular_biases_two = biases(data, point_two)
    box_biases_three, gauss_biases_three, epanechnikov_biases_three, triangular_biases_three = biases(data, point_three)

    plt.figure(22)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), box_biases_one, label='Точка 1')
    plt.plot(np.arange(0.05, 5, 0.05), box_biases_two, label='Точка 2')
    plt.plot(np.arange(0.05, 5, 0.05), box_biases_three, label='Точка 3')
    plt.xlabel('h')
    plt.ylabel('Absolute Bias(h)')
    plt.legend()
    plt.show()

    plt.figure(23)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), gauss_biases_one, label='Точка 1')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_biases_two, label='Точка 2')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_biases_three, label='Точка 3')
    plt.xlabel('h')
    plt.ylabel('Absolute Bias(h)')
    plt.legend()
    plt.show()

    plt.figure(24)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_biases_one, label='Точка 1')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_biases_two, label='Точка 2')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_biases_three, label='Точка 3')
    plt.xlabel('h')
    plt.ylabel('Absolute Bias(h)')
    plt.legend()
    plt.show()

    plt.figure(25)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), triangular_biases_one, label='Точка 1')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_biases_two, label='Точка 2')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_biases_three, label='Точка 3')
    plt.xlabel('h')
    plt.ylabel('Absolute Bias(h)')
    plt.legend()
    plt.show()

    plt.figure(26)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), box_biases_one, label='Прямоугольное окно')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_biases_one, label='Гауссово окна')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_biases_one, label='Окно Епанечникова')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_biases_one, label='Треугольное окно')
    plt.xlabel('h')
    plt.ylabel('Absolute Bias(h)')
    plt.legend()
    plt.show()
    
    plt.figure(27)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), box_biases_two, label='Прямоугольное окно')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_biases_two, label='Гауссово окна')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_biases_two, label='Окно Епанечникова')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_biases_two, label='Треугольное окно')
    plt.xlabel('h')
    plt.ylabel('Absolute Bias(h)')
    plt.legend()
    plt.show()
    
    plt.figure(28)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(np.arange(0.05, 5, 0.05), box_biases_three, label='Прямоугольное окно')
    plt.plot(np.arange(0.05, 5, 0.05), gauss_biases_three, label='Гауссово окна')
    plt.plot(np.arange(0.05, 5, 0.05), epanechnikov_biases_three, label='Окно Епанечникова')
    plt.plot(np.arange(0.05, 5, 0.05), triangular_biases_three, label='Треугольное окно')
    plt.xlabel('h')
    plt.ylabel('Absolute Bias(h)')
    plt.legend()
    plt.show()


