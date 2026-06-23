import numpy as np
import dill


def compare(a, b):

    els = np.sum(a != b)
    total = a.size
    per = 100 * els / total
    return els
