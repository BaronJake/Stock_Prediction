from pandas import DataFrame, read_csv
from yfinance import Ticker
from predictStocks import processStock
from random import sample
from datetime import date

today = date.today().strftime("%b-%d-%Y")


def download_hist(stock_to_download):
    stock_to_download = stock_to_download.upper()
    ticker_obj = Ticker(stock_to_download)
    hist = ticker_obj.history(period="max")
    hist.to_csv("Data/Stocks/"+stock_to_download+".csv")


def sample_stocks(sample_number):
    with open("StockList.txt") as file:
        whole_stocks_list = [x.strip() for x in file.readlines()]
        sampled_list = sample(whole_stocks_list, sample_number)
    return sampled_list


def read_file(stock_ticker):
    # import stock data and drop some columns that weren't useful
    df = read_csv("Data/Stocks/"+stock_ticker+".csv", usecols=[1, 2, 3, 4, 5])
    if len(df) < 500:
        new_stock = sample_stocks(1)
        download_hist(new_stock[0])
        res = read_file(new_stock[0])
    else:
        res = processStock(df.dropna())
    return res


def append_to_file(data):
    DataFrame([data]).to_csv("Data/Outputs/"+today+".csv", mode="a", index=False, header=False)


sampled_stock_list = sample_stocks(50)

for stock in sampled_stock_list:
    download_hist(stock)

print("Done Downloading! Start Predicting!")
output = [
    'Ticker',
    'Day 1',
    'Day 2',
    'Day 3',
    'Day 4',
    'Day 5',
    'Current',
    'Day 1 - Current',
    'Day 5 - Day 1',
    'Eval',
    'Length',
]
append_to_file(output)

for stock in sampled_stock_list:
    pred_results = read_file(stock)
    pred_results.insert(0, stock)
    pred_results.insert(7, pred_results[1]-pred_results[6])
    pred_results.insert(8, pred_results[5]-pred_results[1])
    print(pred_results)
    append_to_file(pred_results)

