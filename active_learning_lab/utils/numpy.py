import numpy as np


def get_class_histogram(y, num_classes, normalize=True, multilabel=False):

    ind, counts = np.unique(y, return_counts=True)
    histogram = np.zeros(num_classes)
    if multilabel:
        for i, c in zip(ind, counts):
            for j in i:
                histogram[j] += c
    else:
        ind_set = set(ind)

        for i, c in zip(ind, counts):
            if i in ind_set:
                histogram[i] = c

    if normalize:
        return histogram / histogram.sum()

    return histogram.astype(int)
