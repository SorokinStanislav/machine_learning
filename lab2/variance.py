import random
import numpy as np


def random_half_data(data):
    half_size = int(len(data) / 2)
    return random.sample(data, half_size)


def variance(naive_density_value):
    return np.std(naive_density_value) ** 2
