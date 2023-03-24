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


# p-values
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    '''
    计算双尾P值
    '''
    if x > mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)


upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

from typing import List
import random


def run_experiment() -> List[bool]:
    '''
    Flip a fair coin 1000 times, True=heads, False=tails
    '''
    return [random.random() < .5 for _ in range(1000)]


def reject_fairness(experiment: list[bool]) -> bool:
    '''
    Using the 5% significance levels
    '''
    # 获取正面向上的数量
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531


# Example: Running an A/B Test
def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    '''

    '''
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma


def a_b_test_statistics(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    '''

    '''
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    # 加上一个很小的数字，避免除以0
    return (p_B - p_A) / (math.sqrt(sigma_A ** 2 + sigma_B ** 2) + 1e-7)


def B(alpha: float, beta: float) -> float:
    '''
    a normalizing constant so that the total probability is 1
    '''
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)


def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
