"""
Paired t-test for significance testing (Hypothesis 1).
"""

from scipy.stats import ttest_rel
import numpy as np
from typing import Dict, List, Tuple

def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = 'two-sided',
) -> Dict[str, float]:
    """
    Perform paired t-test between two sets of scores (e.g., nDCG@10 for two methods).
    Returns t-statistic, p-value, and whether significantly different.
    """
    t_stat, p_value = ttest_rel(scores_a, scores_b)
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': np.mean(np.array(scores_a) - np.array(scores_b)),
    }