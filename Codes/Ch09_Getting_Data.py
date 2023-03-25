with open('email_addresses.txt', 'w') as f:
    f.write("joelgrus@gmail.com\n")
    f.write("joel@m.datasciencester.com\n")
    f.write("joelgrus@m.datasciencester.com\n")


def get_domain(email_address: str) -> str:
    '''Split on  @ and return the last piece'''
    return email_address.lower().split("@")[-1]


from collections import Counter

with open('email_addresses.txt') as f:
    domain_counts = Counter(get_domain(line.strip())
                            for line in f if '@' in line)


import csv

with open('tab_delimited_stock_prices.txt', 'w') as f:
    f.write("""6/20/2014\tAAPL\t90.91
6/20/2014\tMSFT\t41.68
6/20/2014\tFB\t64.5
6/19/2014\tAAPL\t91.86
6/19/2014\tMSFT\t41.51
6/19/2014\tFB\t64.34
""")


def process(date: str, symbol: str, closing_price: float) -> None:
    '''
    假设该函数做了某些事情
    '''
    assert closing_price > 0


with open('tab_delimited_stock_prices.txt') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(date, symbol, closing_price)


with open('colon_delimited_stock_prices.txt', 'w') as f:
    f.write("""date:symbol:closing_price
6/20/2014:AAPL:90.91
6/20/2014:MSFT:41.68
6/20/2014:FB:64.5
""")


with open('colon_delimited_stock_prices.txt') as f:
    colon_reader = csv.DictReader(f, delimiter=':')
    # 每一行都是由header作为key, 的字典，所有字典的长度都是相同的
    for dict_row in colon_reader:
        date = dict_row['date']
        symbol = dict_row['symbol']
        closing_price = float(dict_row['closing_price'])
        process(date, symbol, closing_price)

with open('colon_delimited_stock_prices.txt') as f:
    colon_reader = csv.reader(f, delimiter=':')
    # skip the header row with an initial call to reader.__next__()
    colon_reader.__next__()
    for row in colon_reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(date, symbol, closing_price)


todays_prices = {'AAPL': 90.91, 'MSFT': 41.68, 'FB': 64.5}

with open('comma_delimited_stock_prices.txt', 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    for stock, prices in todays_prices.items():
        csv_writer.writerow([stock, prices])


results = [["test1", "success", "Monday"],
           ["test2", "success, kind of", "Tuesday"],
           ["test3", "failure, kind of", "Wednesday"],
           ["test4", "failure, utter", "Thursday"]]

from bs4 import BeautifulSoup
import requests


url = ("https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')


first_paragraph = soup.find('p')
first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

first_paragraph_id = soup.p['id']  # raises KeyError if no 'id'
first_paragraph_id2 = soup.p.get('id')  # return None if no 'id'


def paragraph_mentions(text: str, keyword: str) -> bool:
    '''
    return True if a <p> inside the text mentions {keyword}
    '''
    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = [p.get_text() for p in soup('p')]
    return any(keyword.lower() in paragraph.lower()
               for paragraph in paragraphs)


import json
serialized = """{
    "title" : "Data Science Book",
    "author" : "Joel Grus",
    "publicationYear" : 2019,
    "topics" : [ "data", "science", "data science"] }"""


deserialized = json.loads(serialized)  # dict


github_user = 'joelgrus'
endpoint = f'https://api.github.com/users/{github_user}/repos'
# List[Dict[str: [str]]]
repos = json.loads(requests.get(endpoint).text)

from collections import Counter
from dateutil.parser import parse

dates = [parse(repo['created_at']) for repo in repos]
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)


last_5_repositories = sorted(repos,
                             key=lambda r: r['pushed_at'],
                             reverse=True)[:5]
last_5_languages = [repo['language'] for repo in last_5_repositories]
