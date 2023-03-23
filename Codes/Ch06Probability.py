def uniform_pdf(x: float) -> float:
    '''
    计算均匀分布的PDF，[a, b] = [0, 1]
    '''
    return 1 if 0 <= x < 1 else 0


def uniform_cdf(x: float) -> float:
    '''
    Returns the probability that a uniform random variable is <= x
    '''
    if x < 0:
        return 0
    elif x < 1:
        return x
    else:
        return 1


import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    '''
    计算正态分布在x处的概率密度函数值
    '''
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2)) / (SQRT_TWO_PI * sigma)


import matplotlib.pyplot as plt


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    '''
    计算正态分布的累计分布函数
    '''
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 1e-5) -> float:
    assert p < 1, '对于正态分布分位数的概率，概率必须小于1'

    # if not standard, compute standard and rescale
    # 递归调用
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z = -10.0
    hi_z = 10.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
    return mid_z

# The Central Limit Theorem
# 中心极限定理


import random


def bernoulli_trial(p: float) -> int:
    '''
    return 1 with probability p and 0 with probability 1-p
    '''
    return 1 if random.random() < p else 0


def binomial(n: int, p: float) -> int:
    '''
    return the sum of n bernoulli(p) trails
    '''
    return sum(bernoulli_trial(p) for _ in range(n))


from collections import Counter


def binomial_histogram(p: float, n: int, num_points: int) -> None:
    '''
    pick points from a Binomial(n, p) and plots their histogram
    '''
    data = [binomial(n, p) for _ in range(num_points)]

    histogram = Counter(data)
    plt.bar([x - .4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            .8, color='.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + .5, mu, sigma) -
          normal_cdf(i - .5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title('Binomial Distributions vs. Normal Approximation')
    plt.show()
