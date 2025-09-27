import numpy as np

def priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """Combines fullness and capacity bonus (v0) with a fit bonus/penalty (v1)."""
    # Avoid division by zero
    bins_remain_cap = np.where(bins_remain_cap == 0, 1e-6, bins_remain_cap)

    remaining_after_add = bins_remain_cap - item
    valid_bins = remaining_after_add >= 0

    priorities = np.full(len(bins_remain_cap), -np.inf)

    if np.any(valid_bins):
        # Fullness score
        fullness = 1 - (remaining_after_add[valid_bins] / bins_remain_cap[valid_bins])
        # Capacity bonus
        capacity_bonus = bins_remain_cap[valid_bins] / np.max(bins_remain_cap)
        # Fit ratio (how well the item fits)
        fit_ratio = item / bins_remain_cap[valid_bins]
        # Good fit bonus
        good_fit_bonus = np.exp(-((fit_ratio - 0.5) ** 2) / 0.08)

        # Combine everything - weighted approach
        priorities[valid_bins] = 0.6 * fullness + 0.2 * capacity_bonus + 0.2 * good_fit_bonus

    return priorities
