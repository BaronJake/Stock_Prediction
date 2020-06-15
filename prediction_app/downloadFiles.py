from pandas import DataFrame, read_csv
from yfinance import Ticker
from prediction_app.predictStocks import processStock
from random import sample
from datetime import date
import logging.config
import sys

today = date.today().strftime("%b-%d-%Y")


def download_hist(stock_to_download):
    try:
        ticker_obj = Ticker(stock_to_download)
        hist = ticker_obj.history(period="max")
        hist.to_csv("data/stocks/"+stock_to_download+".csv")
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to download historical stock data for: %s %s", stock_to_download, str(error))


def get_stock_list():
    try:
        with open("fixtures/stocklist.txt") as file:
            whole_stocks_list = [x.strip() for x in file.readlines()]
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to open file containing list of stocks: %s", str(error))
        raise
    return whole_stocks_list


def sample_stocks(sample_number, stocks_list):
    try:
        sampled_list = sample(stocks_list, sample_number)
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to sample stocks from stocks_list: %s, %s", ','.join(stocks_list), str(error))
        raise
    sampled_list = [sampled_stock.upper() for sampled_stock in sampled_list]
    for sampled_stocks in sampled_list:
        stocks_list.remove(sampled_stocks)
    return sampled_list, stocks_list


def get_new_stock(stocks_list):
    stock_id, new_list = sample_stocks(1, stocks_list)
    download_hist(stock_id[0])
    new_data_frame = read_file(stock_id[0])
    return stock_id[0], new_list, new_data_frame


def validate_stock_file(df):
    logger.info("Number of rows = %s", len(df))
    return len(df) >= 500


def read_file(stock_ticker):
    # import stock data and drop some columns that weren't useful
    try:
        df = read_csv("data/stocks/"+stock_ticker+".csv", usecols=[1, 2, 3, 4, 5])
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to read file for: %s %s", stock_ticker, str(error))
    return df


def append_to_file(data):
    try:
        DataFrame([data]).to_csv("data/outputs/"+today+".csv", mode="a", index=False, header=False)
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to append output data to file for: %s %s", data[0], str(error))

def main():
    number_of_stocks = 100
    all_stocks_list = get_stock_list()
    sampled_stock_list, unused_stocks_list = sample_stocks(number_of_stocks, all_stocks_list)
    logger.info("Sampled stocks")

    stock_number_dict = dict()
    for number, stock in enumerate(sampled_stock_list):
        logger.info("Initial Sampling. Downloading: #%s %s", number, stock)
        stock_number_dict[stock] = number
        download_hist(stock)
    logger.info("Done Downloading! Start Predicting!")

    output = [
        'Ticker',
        'Day 1',
        'Day 2',
        'Day 3',
        'Day 4',
        'Day 5',
        'Day 1 Open',
        'Current',
        'Day 1 Open - Current',
        'Day 5 - Day 1',
        'Day 1 ending - Day 1 open',
        'Eval',
        'Length',
    ]
    append_to_file(output)

    for stock in sampled_stock_list:
        df = read_file(stock)
        logger.info("Reading file for: #%s, %s", stock_number_dict[stock], stock)
        while not validate_stock_file(df):
            logger.info("Needed to get a new stock with more historical data")
            if unused_stocks_list == []:
                logger.info("Ran out of stocks to sample, skipping...")
                continue
            else:
                stock, unused_stocks_list, df = get_new_stock(unused_stocks_list)
            logger.info("New stock: %s", stock)
        pred_results = processStock(df.dropna())
        logger.info("Finished Processing: %s", stock)
        pred_results.insert(0, stock)
        pred_results.insert(8, pred_results[6] - pred_results[7])
        pred_results.insert(9, pred_results[5] - pred_results[1])
        pred_results.insert(10, pred_results[1] - pred_results[6])
        logger.info("Appending results to file: %s", stock)
        append_to_file(pred_results)


if __name__ == '__main__':
    logging.config.fileConfig('logger.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    main()