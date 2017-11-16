import numpy as np


def box_kernel(u):
    if 1 > u > -1:
        return 0.5
    else:
        return 0


def gaussian_kernel(u):
    return 1 / np.math.sqrt(2*np.math.pi) * np.math.exp(-u ** 2 / 2)


def epanechnikov_kernel(u):
    if 1 > u > -1:
        return 0.75 * (1 - u ** 2)
    else:
        return 0


def triangular_kernel(u):
    u_abs = np.math.fabs(u)
    if u_abs < 1:
        return 1 - u_abs
    else:
        return 0