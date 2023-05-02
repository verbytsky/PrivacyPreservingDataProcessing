import yfinance as yf
import pandas as pd


def download_financial_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data


def preprocess_data(data):
    # Drop rows with missing values
    data = data.dropna()

    # Feature selection
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[features]

    return data

def save_processed_data(data, file_name):
    data.to_csv(f'../../data/processed/{file_name}.csv', index=False)
    "data"