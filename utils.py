import numpy as np


def string_to_charlist(s):
    return np.array([ord(c) - ord('a') + 1 for c in s])


def decode_state(state_list, decoder):
    s = ''
    for state in state_list:
        if not decoder[state]:
            s += 'X'
        else:
            s += decoder[state]

    return s


def to_state(s, model):
    d = np.array([string_to_charlist(s)])
    return model.predict(d.T)
