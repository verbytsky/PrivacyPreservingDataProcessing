from data_preparation import download_financial_data, preprocess_data, save_processed_data

if __name__ == '__main__':
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    file_name = 'AAPL_processed_data'

    # Download financial data
    raw_data = download_financial_data(ticker, start_date, end_date)

    # Preprocess data
    processed_data = preprocess_data(raw_data)

    # Save processed data
    save_processed_data(processed_data, file_name)