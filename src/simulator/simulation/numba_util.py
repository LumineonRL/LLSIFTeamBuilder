import numpy as np
from numba import njit

@njit
def _get_combo_multiplier_nb(combo_count: int, combo_bonus_tiers: np.ndarray) -> float:
    """
    Finds the combo multiplier for the current combo count.
    """
    for i in range(combo_bonus_tiers.shape[0]):
        if combo_count + 1 >= combo_bonus_tiers[i, 0]:
            return combo_bonus_tiers[i, 1]
    return 1.0