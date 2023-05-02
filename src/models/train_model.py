import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from diffprivlib.mechanisms import Gaussian
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


def create_data_loaders(data, labels, batch_size):
    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(data_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


class TensorGaussianMechanism:
    def __init__(self, epsilon, delta, sensitivity):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    def randomise(self, tensor):
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(loc=0, scale=sigma, size=tensor.shape)
        noisy_tensor = tensor + noise.astype(tensor.dtype)
        return noisy_tensor


def train_model(model, data_loader, epochs, lr, epsilon, delta):
    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize the custom Gaussian mechanism
    gaussian_mechanism = TensorGaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=1)

    # Training loop
    for epoch in range(epochs):
        for batch in data_loader:
            data, labels = batch
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = loss_fn(output, labels)

            # Backward pass
            loss.backward()

            # Apply the custom Gaussian mechanism to the gradients
            for param in model.parameters():
                grad_numpy = param.grad.detach().numpy()
                noisy_grad = gaussian_mechanism.randomise(grad_numpy)
                param.grad = torch.tensor(noisy_grad)

            # Update the weights
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    delta = 1e-5
    # Load your preprocessed financial data
    data = pd.read_csv('../../data/processed/AAPL_processed_data.csv')
    labels = data['Open']  # Replace with your target column name
    data = data.drop('Open', axis=1)  # Replace with your target column name

    # Define model parameters
    input_size = data.shape[1]
    hidden_size = 64
    output_size = 1
    batch_size = 32
    epochs = 100
    lr = 0.001
    epsilon = 0.1

    # Create the model
    model = SimpleNN(input_size, hidden_size, output_size)

    # Create the data loader
    data_loader = create_data_loaders(data.values, labels.values, batch_size)

    # Train the model with differential privacy
    train_model(model, data_loader, epochs, lr, epsilon, delta)

    # Save the trained model
    model_path = "../models/simple_nn_dp.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")