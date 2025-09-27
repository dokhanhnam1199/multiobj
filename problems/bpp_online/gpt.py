import numpy as np
import random
import math
import scipy
import torch
def priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins_remain_cap: Array of capacities for each bin.

    Return:
        Array of same size as bins_remain_cap with priority score of each bin.
    """
    # Calculate the ratio of item size to bin remaining capacity
    ratios = item / bins_remain_cap

    # Calculate the exponential of the negative ratio
    exp_neg_ratios = np.exp(-ratios)

    # Calculate the square of the ratios
    squared_ratios = ratios ** 2

    # Calculate the logarithm of the ratios (avoid log(0) by adding a small epsilon)
    log_ratios = np.log(ratios + 1e-10)

    # Calculate the difference between item size and bin remaining capacity
    diff = np.abs(item - bins_remain_cap)

    # Combine the features with different weights
    priorities = exp_neg_ratios + 2 * squared_ratios - 0.5 * log_ratios - 3 * diff

    return priorities
