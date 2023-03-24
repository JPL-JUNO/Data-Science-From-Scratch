from Ch04LinearAlgebra import Vector, vector_dot_product
from typing import Callable
from Ch04LinearAlgebra import distance, vector_add, scalar_multiply


def sum_of_squares(v: Vector) -> float:
    '''computes the sum of squared values in v'''

    return vector_dot_product(v, v)


def different_quotient(f: Callable[[float], float],
                       x: float,
                       h: float = 1e-4) -> float:
    ''''''
    return (f(x + h) - f(x - h)) / (2 * h)


def square(x: float) -> float:
    return x * x


def derivative(x: float) -> float:
    return 2 * x


def partial_different_quotient(f: Callable[[Vector], float],
                               v: Vector,
                               i: int, h: float = 1e-4) -> float:
    '''return th i-th partial difference quotient of f at v'''
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    # w = v.copy()
    # x1 = w[i] + h
    # x2 = w[i] - 2 * h
    # f1 = f(x1)
    # f2 = f(x2)
    # return (f1 - f2) / (2 * h)
    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector, h: float = 1e-4) -> float:
    '''逐个计算偏微分
    A major drawback to this “estimate using difference quotients”
    approach is that it’s computationally expensive.
    '''
    return [partial_different_quotient(f, v, i, h) for i in range(len(v))]

# Using the Gradient
# Let’s use gradients to find the minimum among all three-dimensional vectors. We’ll just pick a random starting point
# and then take tiny steps in the opposite direction of the gradient until we reach a
# point where the gradient is very small


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> float:
    '''
    Moves step_size(learning rate) in the gradient direction from v
    用作梯度的改变
    '''
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return vector_add(v, step)


def sum_of_squared_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad


# Minibatch and Stochastic Gradient Descent
from typing import TypeVar, List, Iterator
import random
T = TypeVar('T')  # Can be anything


def minibatch(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    '''Generates batch_size-sized minibatch-es from the dataset'''
    if (len(dataset) % batch_size != 0):
        raise ValueError('数据长度不能被batch_size整除')
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle:  # 是否按照打乱的数据进行训练
        random.shuffle(batch_starts)
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
