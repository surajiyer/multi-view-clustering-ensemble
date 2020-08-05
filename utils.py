# coding: utf-8

"""
    description: Utility functions
    author: Suraj Iyer
"""

import numpy as np
import numba as nb


def rowwise_dissimilarity(values):
    """
    Compare every row with each other row and count number
    of differences along column axis per row pairs.

    Example:
        input: [[1, 2, 3],
                [1, 3, 1],
                [2, 2, 2]]
        output: [[0, 2, 2],
                 [2, 0, 3]
                 [2, 3, 0]]
    """
    return np.sum(values != values[:, None], axis=-1)


def rowwise_cosine_similarity(values):
    """
    Using every pair of rows in :values: as input, compute
    pairwise cosine similarity between each row.

    URL: https://stackoverflow.com/questions/41905029/create-cosine-similarity-matrix-numpy
    """
    norm = (values * values).sum(0, keepdims=True) ** .5
    values = values / norm
    return (values.T @ values)


@nb.njit(parallel=True, fastmath=True)
def convert_to_ultrametric(values):
    """
    Fix triangular inequality within distance matrix (values)
    by converting to ultra-metric by ensuring the following
    condition: d_{ij} = min(d_{ij}, max(d_{ik}, d_{kj}))

    Parameters:
    ------------
    values: np.ndarray
        2D square distance matrix.

    Returns:
    --------
    np.ndarray
        Ultrametrified distance matrix.
    """
    values = np.atleast_2d(values)
    result = np.full(values.shape, 1.)
    for i in nb.prange(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            result[i, j] = result[j, i] = min(np.min(
                np.fmax(values[i], values[j])), values[i, j])
    return result
