# Statistics
import math
from ch04 import sum_of_squares
from ch04 import vector_dot_product
from typing import List
from collections import Counter


def stat_mean(xs: List[float]) -> float:
    ''''''
    return sum(xs) / len(xs)


def _median_odd(xs: List[float]) -> float:
    '''
    '''
    return sorted(xs)[len(xs) // 2]


def _median_even(xs: List[float]) -> float:
    '''
    '''
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2


def stat_median(v: List[float]) -> float:
    '''
    '''
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


def stat_quantile(xs: List[float], p: float) -> float:
    '''
    '''
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def stat_mode(x: List[float]) -> float:
    '''
    '''
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)


def de_mean(xs: List[float]) -> List[float]:
    '''
    计算偏差，样本值-均值
    '''
    x_bar = stat_mean(xs)
    return [x - x_bar for x in xs]


def stat_variance(xs: List[float]) -> float:
    '''
    '''
    assert len(xs) >= 2, 'variance requires at least two elements'

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(xs: List[float]) -> float:
    '''
    '''
    return math.sqrt(stat_variance(xs))


def inter_quartile_range(xs: List[float]) -> float:
    '''
    '''
    return stat_quantile(xs, .75) - stat_quantile(xs, .25)

# Correlation


def stat_covariance(xs: List[float], ys: List[float]) -> float:
    '''
    '''
    assert len(xs) == len(ys), 'xs and ys must have same number of elements'

    return vector_dot_product(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


def correlation(xs: List[float], ys: List[float]) -> float:
    '''
    '''
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return stat_covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0
