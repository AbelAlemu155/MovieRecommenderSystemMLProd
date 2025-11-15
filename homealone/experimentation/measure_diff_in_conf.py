
import numpy as np
from scipy.stats import ttest_ind


def compute_confidence(t_measures, c_measures):
    t_stat, p_value = ttest_ind(t_measures, c_measures, equal_var=False)

    print(f"tstat, {t_stat}")
    print(f"p_value, {p_value}")
    return t_stat, p_value