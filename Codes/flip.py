import random
from collections import Counter


def flip_coin(p: float = .5) -> str:
    '''
    计算一次抛硬币正反面，
    1：正面
    0：反面
    '''
    return 1 if random.random() < p else 0


def binomial(n: int, p: float = .5) -> float:
    '''
    计算n次抛硬币正面次数
    '''
    return sum(flip_coin(p) for _ in range(n))


ret = [binomial(100) for _ in range(100)]
