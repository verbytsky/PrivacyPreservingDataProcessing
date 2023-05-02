import pandas as pd
from diffprivlib.tools import mean, sum

## Function to calculate the bounds of the data
def calculate_bounds(data):
    lower_bounds = data.min().tolist()
    upper_bounds = data.max().tolist()

    return lower_bounds, upper_bounds

## Function to apply differential privacy to the data
def apply_differential_privacy(data, epsilon):
    # Calculate the data bounds
    lower_bounds, upper_bounds = calculate_bounds(data)
    bounds = list(zip(lower_bounds, upper_bounds))

    # Compute differentially private statistics
    dp_mean = [mean(data[col], epsilon=epsilon, bounds=bounds[i]) for i, col in enumerate(data.columns)]
    dp_sum = [sum(data[col], epsilon=epsilon, bounds=bounds[i]) for i, col in enumerate(data.columns)]
    dp_count = len(data)

    # Get the differentially private results
    dp_data = pd.DataFrame({'Mean': dp_mean, 'Total': dp_sum, 'Count': [dp_count] * len(dp_mean)}, index=data.columns)

    return dp_data