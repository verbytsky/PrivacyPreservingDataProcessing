from feature_extraction import apply_differential_privacy
import pandas as pd

if __name__ == '__main__':
    file_name = 'AAPL_processed_data'
    epsilon = 0.1

    ## Load processed data
    data = pd.read_csv(f'../../data/processed/{file_name}.csv')

    ## Apply differential privacy to the data
    dp_data = apply_differential_privacy(data, epsilon)

    ## Print the differentially private results
    print(dp_data)