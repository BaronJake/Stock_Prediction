import pandas as pd
import yfinance as yf
from predictStocks import processStock
from random import sample


def download_hist(stock_to_download):
    stock_to_download = stock_to_download.upper()
    ticker_obj = yf.Ticker(stock_to_download)
    hist = ticker_obj.history(period="max")
    hist.to_csv("C:/Users/jmper/OneDrive/Documents/Data Sets/Stocks/"+stock_to_download+".csv")


def read_file(stock_ticker):
    # import stock data and drop some columns that weren't useful
    df = pd.read_csv("C:/Users/jmper/OneDrive/Documents/Data Sets/Stocks/"+stock_ticker+".csv", usecols=[1, 2, 3, 4, 5])
    if len(df) < 500:
        new_stock = sample_stocks(1)
        download_hist(new_stock[0])
        res = read_file(new_stock[0])
    else:
        res = processStock(df.dropna())
    return res


def sample_stocks(sample_number):
    with open("StockList.txt") as file:
        whole_stocks_list = [x.strip() for x in file.readlines()]
        sampled_list = sample(whole_stocks_list, sample_number)
    return sampled_list


sampled_stock_list = sample_stocks(25)

for stock in sampled_stock_list:
    download_hist(stock)

print("Done Downloading! Start Predicting!")
output = [['Ticker', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Current', 'Day 1 - Current', 'Eval', 'Length']]

for stock in sampled_stock_list:
    pred_results = read_file(stock)
    pred_results.insert(0, stock)
    pred_results.insert(7, pred_results[1]-pred_results[6])
    print(pred_results)
    output.append(pred_results)
pd.DataFrame(output).to_csv('Output.csv')

