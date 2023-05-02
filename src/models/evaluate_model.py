import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.models.train_model import SimpleNN, create_data_loaders
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch
            output = model(data)
            y_true.extend(labels.numpy().flatten())
            y_pred.extend(output.numpy().flatten())

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    return mae, mse


def split_data(data, labels, test_size):
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Load preprocessed financial data
    data = pd.read_csv('../../data/processed/AAPL_processed_data.csv')
    labels = data['Open']  # Replace with your target column name
    data = data.drop('Open', axis=1)  # Replace with your target column name

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(data.values, labels.values, test_size=0.2)

    # Define model parameters
    input_size = data.shape[1]
    hidden_size = 64
    output_size = 1
    batch_size = 32

    # Load the trained model
    model = SimpleNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('simple_nn_dp.pt'))

    # Create data loaders for validation set
    val_data_loader = create_data_loaders(X_val, y_val, batch_size)

    # Evaluate the model on the validation set
    mae, mse = evaluate_model(model, val_data_loader)
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
