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


from dateutil.parser import parse


def prase_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol=symbol, date=parse(date).date(), closing_price=float(closing_price))


stock = prase_row(['MSFT', "2018-12-14", "106.03"])

assert stock.symbol == 'MSFT'
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03


from typing import Optional
import re


def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row
    if not re.match(r"^[A-Z]+$", symbol):
        return None
    try:
        date = parse(date_).date()
    except ValueError:
        return None

    try:
        closing_price = float(closing_price_)
    except ValueError:
        return None
    return StockPrice(symbol, date, closing_price)


assert try_parse_row(['MSFT0', '2018-12-14', '106.03']) is None
assert try_parse_row(['MSFT', '2018-12--14', '106.03']) is None
assert try_parse_row(['MSFT', '2018-12-14', 'x']) is None

assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock


import csv

data: List[StockPrice] = []

# 这个数据太少，换下面的代码块进行读取
# with open("comma_delimited_stock_prices.csv", "r") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         maybe_stock = try_parse_row(row)
#         if maybe_stock is None:
#             print('skipping invalid row: {}'.format(row))
#         else:
#             data.append(maybe_stock)


with open('stocks.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = [[row['Symbol'], row['Date'], row['Close']]
            for row in reader]

maybe_data = [try_parse_row(row) for row in rows]
assert maybe_data
assert all(sp is not None for sp in maybe_data)

data = [sp for sp in maybe_data if sp is not None]

max_aapl_price = max(stock_price.closing_price
                     for stock_price in data
                     if stock_price.symbol == 'AAPL')

from collections import defaultdict

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))
for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price

from typing import List
from collections import defaultdict
prices: Dict[str, List[StockPrice]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)

prices = {symbol: sorted(symbol_prices)
          for symbol, symbol_prices in prices.items()}


def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    """计算涨跌幅

    Args:
        yesterday (StockPrice): _description_
        today (StockPrice): _description_

    Returns:
        float: _description_
    """
    return today.closing_price / yesterday.closing_price - 1


class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float


def day_over_day_changes(prices: List[StockPrice]) -> List[DailyChange]:
    """假设价格都是都一个symbol并且是排序了的

    Args:
        prices (List[StockPrice]): _description_

    Returns:
        List[DailyChange]: _description_
    """
    return [DailyChange(symbol=today.symbol, date=today.date, pct_change=pct_change(yesterday, today))
            for yesterday, today in zip(prices, prices[1:])]


all_changes = [change for symbol_prices in prices.values()
               for change in day_over_day_changes(symbol_prices)]


max_change = max(all_changes, key=lambda change: change.pct_change)

assert max_change.symbol == 'AAPL'
assert max_change.date == datetime.date(1997, 8, 6)
assert .33 < max_change.pct_change < .34


change_by_month: Dict[int, List[DailyChange]] = {
    month: [] for month in range(1, 13)}

for change in all_changes:
    change_by_month[change.date.month].append(change)

avg_daily_change = {
    month: sum(change.pct_change for change in changes) / len(changes)
    for month, changes in change_by_month.items()
}


assert avg_daily_change[10] == max(avg_daily_change.values())

# Rescaling

from Ch04LinearAlgebra import distance, vector_mean

# height in inches
a_to_b = distance([63, 150], [67, 160])
a_to_c = distance([63, 150], [70, 171])
b_to_c = distance([67, 160], [70, 171])

# height in centimeters
a_to_b = distance([160, 150], [170.2, 160])  # 14.28
a_to_c = distance([160, 150], [177.8, 171])  # 27.53
b_to_c = distance([170.2, 160], [177.8, 171])  # 13.37


from typing import Tuple
from Ch05Statistics import standard_deviation


def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """计算 each position 的均值和标准差

    Args:
        data (List[Vector]): _description_

    Returns:
        Tuple[Vector, Vector]: _description_
    """
    dim = len(data[0])

    means = vector_mean(data)  # 计算的是列
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]
    return means, stdevs


vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)
assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]


def rescale(data: List[Vector]) -> List[Vector]:
    """_summary_

    Args:
        data (List[Vector]): _description_

    Returns:
        List[Vector]: _description_
    """
    # dim 指的是向量中的长度，不是list的长度
    dim = len(data[0])
    means, stdevs = scale(data)

    # make a copy of each vector
    rescaled = [v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            # 方差等于0的不做任何的操作
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
    return rescaled


means, stdevs = scale(rescale(vectors))
assert means == [0, 0, 1]
assert stdevs == [1, 1, 0]

from typing import List
import tqdm


def primes_up_to(n: int) -> List[int]:
    primes = [2]

    with tqdm.trange(3, n) as t:
        for i in t:
            i_is_prime = not any(i % p == 0 for p in primes)
            if i_is_prime:
                primes.append(i)
            t.set_description('{} primes'.format(len(primes)))
    return primes

# Dimensionality Reduction


from Ch04LinearAlgebra import vector_subtract


def de_mean(data: List[Vector]) -> List[Vector]:
    """Recenters the data to have mean 0 in every dimension

    Args:
        data (List[Vector]): _description_

    Returns:
        List[Vector]: _description_
    """
    mean = vector_mean(data)
    return [vector_subtract(vector, mean) for vector in data]


from Ch04LinearAlgebra import magnitude


def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]


from Ch04LinearAlgebra import vector_dot_product


def directional_variance(data: List[Vector], w: Vector) -> float:
    """return the variance of x in th direction of w

    Args:
        data (List[Vector]): _description_
        w (Vector): _description_

    Returns:
        float: _description_
    """
    w_dir = direction(w)
    return sum(vector_dot_product(v, w_dir) ** 2 for v in data)


def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """the gradient of directional variance with respect to w

    Args:
        data (List[Vector]): _description_
        w (Vector): _description_

    Returns:
        Vector: _description_
    """
    w_dir = direction(w)
    return [sum(2 * vector_dot_product(v, w_dir) * v[i] for v in data)
            for i in range(len(w))]


from Ch08_gradient_descent import gradient_step


def first_principal_component(data: List[Vector],
                              n: int = 100,
                              step_size: float = .1) -> Vector:
    """start with a random guess

    Args:
        data (List[Vector]): _description_
        n (int, optional): _description_. Defaults to 100.
        step_size (float, optional): _description_. Defaults to .1.

    Returns:
        Vector: _description_
    """
    guess = [1.0 for _ in data[0]]
    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description('dv: {:.3f}'.format(dv))
    return direction(guess)


pca_data = [
    [20.9666776351559, -13.1138080189357],
    [22.7719907680008, -19.8890894944696],
    [25.6687103160153, -11.9956004517219],
    [18.0019794950564, -18.1989191165133],
    [21.3967402102156, -10.8893126308196],
    [0.443696899177716, -19.7221132386308],
    [29.9198322142127, -14.0958668502427],
    [19.0805843080126, -13.7888747608312],
    [16.4685063521314, -11.2612927034291],
    [21.4597664701884, -12.4740034586705],
    [3.87655283720532, -17.575162461771],
    [34.5713920556787, -10.705185165378],
    [13.3732115747722, -16.7270274494424],
    [20.7281704141919, -8.81165591556553],
    [24.839851437942, -12.1240962157419],
    [20.3019544741252, -12.8725060780898],
    [21.9021426929599, -17.3225432396452],
    [23.2285885715486, -12.2676568419045],
    [28.5749111681851, -13.2616470619453],
    [29.2957424128701, -14.6299928678996],
    [15.2495527798625, -18.4649714274207],
    [26.5567257400476, -9.19794350561966],
    [30.1934232346361, -12.6272709845971],
    [36.8267446011057, -7.25409849336718],
    [32.157416823084, -10.4729534347553],
    [5.85964365291694, -22.6573731626132],
    [25.7426190674693, -14.8055803854566],
    [16.237602636139, -16.5920595763719],
    [14.7408608850568, -20.0537715298403],
    [6.85907008242544, -18.3965586884781],
    [26.5918329233128, -8.92664811750842],
    [-11.2216019958228, -27.0519081982856],
    [8.93593745011035, -20.8261235122575],
    [24.4481258671796, -18.0324012215159],
    [2.82048515404903, -22.4208457598703],
    [30.8803004755948, -11.455358009593],
    [15.4586738236098, -11.1242825084309],
    [28.5332537090494, -14.7898744423126],
    [40.4830293441052, -2.41946428697183],
    [15.7563759125684, -13.5771266003795],
    [19.3635588851727, -20.6224770470434],
    [13.4212840786467, -19.0238227375766],
    [7.77570680426702, -16.6385739839089],
    [21.4865983854408, -15.290799330002],
    [12.6392705930724, -23.6433305964301],
    [12.4746151388128, -17.9720169566614],
    [23.4572410437998, -14.602080545086],
    [13.6878189833565, -18.9687408182414],
    [15.4077465943441, -14.5352487124086],
    [20.3356581548895, -10.0883159703702],
    [20.7093833689359, -12.6939091236766],
    [11.1032293684441, -14.1383848928755],
    [17.5048321498308, -9.2338593361801],
    [16.3303688220188, -15.1054735529158],
    [26.6929062710726, -13.306030567991],
    [34.4985678099711, -9.86199941278607],
    [39.1374291499406, -10.5621430853401],
    [21.9088956482146, -9.95198845621849],
    [22.2367457578087, -17.2200123442707],
    [10.0032784145577, -19.3557700653426],
    [14.045833906665, -15.871937521131],
    [15.5640911917607, -18.3396956121887],
    [24.4771926581586, -14.8715313479137],
    [26.533415556629, -14.693883922494],
    [12.8722580202544, -21.2750596021509],
    [24.4768291376862, -15.9592080959207],
    [18.2230748567433, -14.6541444069985],
    [4.1902148367447, -20.6144032528762],
    [12.4332594022086, -16.6079789231489],
    [20.5483758651873, -18.8512560786321],
    [17.8180560451358, -12.5451990696752],
    [11.0071081078049, -20.3938092335862],
    [8.30560561422449, -22.9503944138682],
    [33.9857852657284, -4.8371294974382],
    [17.4376502239652, -14.5095976075022],
    [29.0379635148943, -14.8461553663227],
    [29.1344666599319, -7.70862921632672],
    [32.9730697624544, -15.5839178785654],
    [13.4211493998212, -20.150199857584],
    [11.380538260355, -12.8619410359766],
    [28.672631499186, -8.51866271785711],
    [16.4296061111902, -23.3326051279759],
    [25.7168371582585, -13.8899296143829],
    [13.3185154732595, -17.8959160024249],
    [3.60832478605376, -25.4023343597712],
    [39.5445949652652, -11.466377647931],
    [25.1693484426101, -12.2752652925707],
    [25.2884257196471, -7.06710309184533],
    [6.77665715793125, -22.3947299635571],
    [20.1844223778907, -16.0427471125407],
    [25.5506805272535, -9.33856532270204],
    [25.1495682602477, -7.17350567090738],
    [15.6978431006492, -17.5979197162642],
    [37.42780451491, -10.843637288504],
    [22.974620174842, -10.6171162611686],
    [34.6327117468934, -9.26182440487384],
    [34.7042513789061, -6.9630753351114],
    [15.6563953929008, -17.2196961218915],
    [25.2049825789225, -14.1592086208169]
]
