import math
from typing import List, Tuple, Callable

Vector = List[float]


def vector_add(v: Vector, w: Vector) -> Vector:
    '''
    '''
    assert len(v) == len(w), 'Vector must be the same length'

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v: Vector, w: Vector) -> Vector:
    '''
    '''
    assert len(v) == len(w), 'Vector must be the same length'

    return [v_i - w_i for v_i, w_i in zip(v, w)]


assert vector_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert vector_subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def vector_sum(vectors: List[Vector]) -> Vector:
    '''
    sum all corresponding elements
    '''

    assert vectors, 'no vectors provided'

    num_elements = len(vectors[0])

    assert all(len(v) == num_elements for v in vectors), 'different size!'

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


def scalar_multiply(c: float, v: Vector) -> Vector:
    '''
    '''
    return [c * v_i for v_i in v]


assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


def vector_mean(vectors: List[Vector]) -> Vector:
    '''
    '''
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


def vector_dot_product(v: Vector, w: Vector) -> float:
    '''
    '''
    assert len(v) == len(w), 'vectors must be same length'

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert vector_dot_product([1, 2, 3], [4, 5, 6]) == 32


def sum_of_squares(v: Vector) -> float:
    '''
    '''
    return vector_dot_product(v, v)


assert sum_of_squares([1, 2, 3]) == 14


def magnitude(v: Vector) -> float:
    '''
    '''
    return math.sqrt(sum_of_squares(v))


assert magnitude([3, 4]) == 5


def squared_distance(v: Vector, w: Vector) -> float:
    '''
    计算两个向量之间平方和
    '''
    return sum_of_squares(vector_subtract(v, w))


def distance1(v: Vector, w: Vector) -> float:
    '''
    计算两个向量之间的距离
    '''
    return math.sqrt(sum_of_squares(v, w))


def distance(v: Vector, w: Vector) -> float:
    '''
    与上面的distance等价，但是更简洁
    '''
    return magnitude(vector_subtract(v, w))

# Matrices


Matrix = List[List[float]]

A = [[1, 2, 3],
     [4, 5, 6]]

B = [[1, 2],
     [3, 4],
     [5, 6]]


def shape(A: Matrix) -> Tuple[int, int]:
    '''
    计算矩阵的形状，行数和列数
    '''
    num_rows = len(A)
    num_cols = len(A[0])
    return num_rows, num_cols


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)


def get_row(A: Matrix, i: int) -> Vector:
    '''
    获取矩阵的某行
    '''
    return A[i]


def get_column(A: Matrix, j: int) -> Vector:
    '''
    获取矩阵的某列，
    由于列表是从0开始的，因此获取的是j+1列
    '''
    return [A_i[j] for A_i in A]


def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    '''
    '''
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


def identity_matrix(n: int) -> Matrix:
    '''
    '''
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


assert identity_matrix(3) == [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1],]
