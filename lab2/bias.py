import numpy as np
import scipy.stats as sp


def true_pdf(x):
    return 1/6 * sp.norm.pdf(x) + 1/2 * sp.norm.pdf(x, 5, 1.5) + 1/3 * sp.norm.pdf(x, 8.5, 4)


