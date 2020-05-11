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


def get_stock_list():
    with open("StockList.txt") as file:
        whole_stocks_list = [x.strip() for x in file.readlines()]
    return whole_stocks_list


def sample_stocks(sample_number, stocks_list):
    sampled_list = sample(stocks_list, sample_number)
    for sampled_stocks in sampled_list:
        stocks_list.remove(sampled_stocks)
    return sampled_list, stocks_list


def get_new_stock(stocks_list):
    stock_id, new_list = sample_stocks(1, stocks_list)
    download_hist(stock_id[0])
    new_data_frame = read_file(stock[0])
    return stock_id[0], new_list, new_data_frame


def validate_stock_file(df):
    return len(df) >= 500


def read_file(stock_ticker):
    # import stock data and drop some columns that weren't useful
    df = read_csv("Data/Stocks/"+stock_ticker+".csv", usecols=[1, 2, 3, 4, 5])
    return df


def append_to_file(data):
    DataFrame([data]).to_csv("Data/Outputs/"+today+".csv", mode="a", index=False, header=False)


all_stocks_list = get_stock_list()
sampled_stock_list, unused_stocks_list = sample_stocks(50, all_stocks_list)

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
    'Day 1 Open',
    'Current',
    'Day 1 - Current',
    'Day 5 - Day 1',
    'Eval',
    'Length',
]
append_to_file(output)

for stock in sampled_stock_list:
    df = read_file(stock)
    while not validate_stock_file(df):
        stock, unused_stocks_list, df = get_new_stock(unused_stocks_list)
    pred_results = processStock(df.dropna())
    pred_results.insert(0, stock)
    pred_results.insert(8, pred_results[1]-pred_results[7])
    pred_results.insert(9, pred_results[5]-pred_results[1])
    append_to_file(pred_results)

