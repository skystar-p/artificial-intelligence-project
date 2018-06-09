import numpy as np


def string_to_charlist(s):
    return np.array([ord(c) - ord('a') + 1 for c in s])
