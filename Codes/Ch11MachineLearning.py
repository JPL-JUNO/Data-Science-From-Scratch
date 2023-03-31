import random
from typing import TypeVar, List, Tuple

X = TypeVar('X')


def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """split data into fractions [prob, 1-prob]

    Args:
        data (List[X]): _description_
        prob (float): _description_

    Returns:
        Tuple[List[X], List[X]]: _description_
    """
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]


data = [n for n in range(1000)]
train, test = split_data(data, .75)

assert len(train) == 750
assert len(test) == 250

assert sorted(train + test) == data


Y = TypeVar('Y')


def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float = .25) -> Tuple[List[X], List[X],
                                                     List[Y], List[Y]]:
    """_summary_

    Args:
        xs (List[X]): _description_
        ys (List[Y]): _description_
        test_pct (float, optional): _description_. Defaults to .25.

    Returns:
        Tuple[List[X], List[X], List[Y], List[Y]]: _description_
    """
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])


xs = [x for x in range(1000)]
ys = [2 * x for x in xs]

x_train, x_test, y_train, y_test = train_test_split(xs, ys, .25)

# Check that the proportions are correct
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

# Check that the corresponding data points are paired correctly
assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """_summary_

    Args:
        tp (int): _description_
        fp (int): _description_
        fn (_type_): _description_

    Returns:
        float: _description_
    """
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


assert accuracy(70, 4930, 13930, 981070) == .98114


def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)


assert recall(70, 4930, 13930, 981070) == 0.005


def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)


assert precision(70, 4930, 13930, 981070) == 0.014
