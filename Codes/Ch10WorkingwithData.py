from typing import List, Dict
from collections import Counter
import math
import matplotlib.pyplot as plt
from Ch06Probability import inverse_normal_cdf
import random
from Ch04LinearAlgebra import Matrix, Vector, make_matrix
from Ch05Statistics import correlation


def bucketize(point: float, bucket_size: float) -> float:
    """Floor the point to the next lower multiple of bucket_size

    Args:
        point (float): _description_
        bucket_size (float): _description_

    Returns:
        float: _description_
    """
    return bucket_size * math.floor(point / bucket_size)


def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many in each bucket

    Args:
        points (List[float]): _description_
        bucket_size (float): _description_

    Returns:
        Dict[float, int]: _description_
    """
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points: List[float], bucket_size: float, title: str = ''):
    """_summary_

    Args:
        points (List[float]): _description_
        bucket_size (float): _description_
        title (str, optional): _description_. Defaults to ''.
    """
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)


def random_normal() -> float:
    """_summary_

    Returns:
        float: _description_
    """
    return inverse_normal_cdf(random.random())


def correlation_matrix(data: List[Vector]) -> Matrix:
    """_summary_

    Args:
        data (List[Vector]): _description_

    Returns:
        Matrix: _description_
    """
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])

    return make_matrix(len(data), len(data), correlation_ij)


from typing import NamedTuple
import datetime


class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """
        这是一个类， 因此可以添加它的方法
        """
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']


price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03
assert price.is_high_tech()


from dataclasses import dataclass


@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']
