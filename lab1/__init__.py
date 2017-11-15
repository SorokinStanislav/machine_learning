import csv
import random

from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from lab1.binary_classification_measure import binary_classification_measure

with open('resources/data_v1-08.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)
    prediction = []
    inverted_actual = []
    actual = []
    x1_positive = []
    x2_positive = []
    x1_negative = []
    x2_negative = []

    # determine actual and predicted values from csv
    for row in reader:
        if int(row[2]) > 0:
            x1_positive.append(float(row[0]))
            x2_positive.append(float(row[1]))
        else:
            x1_negative.append(float(row[0]))
            x2_negative.append(float(row[1]))
        if float(row[3]) >= 0:
            actual_elem = 1
        else:
            actual_elem = -1
        actual.extend([int(row[2])])
        prediction.extend([actual_elem])
        inverted_actual.extend([int(row[2]) * (-1)])

    # data visualization
    plt.figure(1)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(x1_negative, x2_negative, marker="o", color='r', label="Положительные")
    plt.scatter(x1_positive, x2_positive, marker="o", color='g', label="Отрицательные")
    plt.xlabel("x2")
    plt.ylabel("x1")
    plt.legend()

    # confusion matrix
    direct_matrix = confusion_matrix(actual, prediction)
    inverted_matrix = confusion_matrix(inverted_actual, prediction)

    # compute measures
    direct_measures = binary_classification_measure(direct_matrix, True)
    invert_measures = binary_classification_measure(inverted_matrix, True)

    # get only lines with positive label from csv
    f.seek(0)
    reader = csv.reader(f)
    next(reader, None)
    positive_lines = [row for row in reader if int(row[2]) > 0]

    # get only lines with negative label from csv
    f.seek(0)
    reader = csv.reader(f)
    next(reader, None)
    negative_lines = [row for row in reader if int(row[2]) < 0]

    # general information about a sample
    initial_volume = direct_measures['volume']
    positive_part = direct_measures['positive_part']

    volumes = range(4, initial_volume + 1)
    sensitivities = []
    specificities = []

    # get random positive and negative lines depend on volume
    for n in volumes:
        actual = []
        prediction = []
        number_of_positives = int(round(n * positive_part, 0))
        number_of_negatives = n - number_of_positives
        positives = random.sample(positive_lines, number_of_positives)
        negatives = random.sample(negative_lines, number_of_negatives)
        data_set = positives + negatives

        for row in data_set:
            if float(row[3]) >= 0:
                actual_elem = 1
            else:
                actual_elem = -1
            actual.extend([int(row[2])])
            prediction.extend([actual_elem])

        matrix = confusion_matrix(actual, prediction)
        measure = binary_classification_measure(matrix)
        sensitivities.append(str(float(measure['sensitivity'])))
        specificities.append(str(float(measure['specificity'])))

    # draw figures
    plt.figure(3)
    plt.plot(volumes, sensitivities)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('N')
    plt.ylabel('Sensitivity')
    plt.show()

    plt.figure(4)
    plt.plot(volumes, specificities)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('N')
    plt.ylabel('Specificity')
    plt.show()

exit(0)
