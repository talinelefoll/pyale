import numpy as np
import pandas as pd
import pprint
from pyale.ale_1d_calcu import calculate_ale_1d

# ALE normalization function
def create_ale_y_norm_function(y_vals):
    centred_y = y_vals - np.median(y_vals)
    pre_median = np.max(centred_y[centred_y < 0])
    post_median = np.min(centred_y[centred_y > 0])

    def norm_ale_y(ale_y):
        norm_ale_y = np.where(
            (ale_y >= pre_median) & (ale_y <= post_median), 0,
            np.where(
                ale_y < 0, -np.searchsorted(np.sort(-centred_y[centred_y <= 0]), -ale_y) / (2 * len(centred_y)),
                np.where(
                    ale_y == 0, 0,
                    np.searchsorted(np.sort(centred_y[centred_y >= 0]), ale_y) / (2 * len(centred_y))
                )
            )
        )
        return norm_ale_y * 100

    return norm_ale_y

# Calculate ALE statistics
def ale_stats(ale_y, ale_n, y_vals=None, ale_y_norm_fun=None, zeroed_ale=False):
    assert y_vals is not None or ale_y_norm_fun is not None, \
        'Either y_vals or ale_y_norm_fun must be provided.'

    if not zeroed_ale:
        raise ValueError('Zeroed ALE required for now.')

    # Remove any NaN ale_y values and corresponding ale_n
    mask = ~np.isnan(ale_y)
    ale_y = ale_y[mask]
    ale_n = ale_n[mask]

    # Internal function for ALED and NALED calculation
    def aled_score(y, n):
        return np.abs(y * n).sum() / n.sum()

    # Average effect in units of y
    aled = aled_score(ale_y, ale_n)

    # Minimum negative and positive effects in units of y
    aler = (np.min(ale_y), np.max(ale_y))

    # Normalized scores
    if ale_y_norm_fun is None:
        ale_y_norm_fun = create_ale_y_norm_function(y_vals)
    norm_ale_y = ale_y_norm_fun(ale_y)

    # NALED scale is 0 to 100, representing equivalent average percentile effect
    naled = aled_score(norm_ale_y, ale_n)

    # Scale is -50 to +50, representing lowest and highest percentile deviations from the median
    naler = (np.min(norm_ale_y), np.max(norm_ale_y))

    return {
        'aled': aled,
        'aler_min': aler[0],
        'aler_max': aler[1],
        'naled': naled,
        'naler_min': naler[0],
        'naler_max': naler[1]
    }