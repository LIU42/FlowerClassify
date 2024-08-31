import numpy as np


def parse_outputs(outputs):
    probability_outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=0)

    return np.argmax(probability_outputs), probability_outputs
