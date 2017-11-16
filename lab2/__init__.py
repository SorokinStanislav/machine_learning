import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from lab2.task1 import task1
from lab2.task2 import task2


def read_data():
    with open('resources/data_v2-08.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        x = []
        for row in reader:
            x.append(float(row[0]))
        return x


def visualize_input_data(input_data):
    plt.figure(1)
    ax = plt.axes()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.legend()
    plt.plot(input_data, len(input_data) * [1], "x", marker='x', color='red')
    plt.show()


data = read_data()
visualize_input_data(data)
task1(data)
task2(data)

exit(0)


