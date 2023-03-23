from typing import Tuple
import math
from Ch06Probability import normal_cdf
from Ch06Probability import inverse_normal_cdf


def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    '''
    returns mu and sigma corresponding to a Binomial(n, p)
    '''
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma


# 主要是计算概率，各种概率，大于，小于，区间，区间外
normal_probability_below = normal_cdf


def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1.0) -> float:
    '''
    the probability that an N(mu, sigma) is greater than lo.
    '''
    return 1 - normal_cdf(lo, mu, sigma)


def normal_probability_between(lo: float, hi: float,
                               mu: float = 0, sigma: float = 1) -> float:
    '''
     the probability that an N(mu, sigma) is between lo. and hi.
    '''
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


def normal_probability_outside(lo: float, hi: float,
                               mu: float = 0, sigma: float = 1) -> float:
    '''
    The probability that an N(mu, sigma) is not between lo and hi.
    '''
    return 1 - normal_probability_between(lo, hi, mu, sigma)


def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    '''
    计算下分位数（左）
    '''
    assert probability < 1
    return inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    '''
    计算上分位数（右）
    '''
    assert probability < 1
    return inverse_normal_cdf(1 - probability, mu, sigma)


def normal_two_sided_bounds(probability: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
    '''
    计算两侧分位数
    '''
    assert probability < 1
    tail_probability = (1 - probability) / 2
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound
