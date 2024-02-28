# Import the prepare_data function from Data_Preparation.py
from Data_Preparation import prepare_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Define your list of tickers and date range
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
start_date = '2023-02-27'
end_date = '2024-02-27'

# Call the prepare_data function to get the data loader and daily returns
data_loader, daily_returns = prepare_data(tickers, start_date, end_date)

# Now you can proceed with your modeling using the data_loader and daily_returns

# Define the portfolio optimization model
class Portfolio(nn.Module):
    def __init__(self, input_size, output_size):
        super(Portfolio, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)  # Use Softmax to ensure allocations sum up to 1
        )

    def forward(self, x):
        return self.network(x)

# Define the loss function
def loss_function(predictions, targets, lambda_reg=0.5):
    portfolio_return = torch.mean(torch.sum(predictions * targets, dim=1))
    portfolio_variance = torch.var(torch.sum(predictions * targets, dim=1))
    total_loss = -portfolio_return + lambda_reg * portfolio_variance
    return total_loss

# Train the model
def train_model(model, data_loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Set hyperparameters
input_size = len(tickers)  # Assuming tickers is the list of stock tickers
output_size = len(tickers)  # Output size is same as input size for portfolio optimization
learning_rate = 0.001

# Initialize model, optimizer, and data loader
model = Portfolio(input_size=input_size, output_size=output_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, data_loader, optimizer)


# Get the asset allocation from the optimized portfolio
with torch.no_grad():
    allocation = model(data_loader.dataset.tensors[0]).numpy()[-1]  # Get allocation for the last batch

# Plot the allocation
plt.figure(figsize=(8, 6))
plt.bar(range(len(tickers)), allocation, tick_label=tickers)
plt.xlabel('Asset')
plt.ylabel('Allocation')
plt.title('Portfolio Allocation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Assuming 'allocation' contains the weights allocated by your portfolio optimization model
# 'daily_returns' contains the daily returns of each asset

portfolio_daily_return = np.sum(allocation * daily_returns, axis=1)
cumulative_return_portfolio = np.cumprod(1 + portfolio_daily_return) - 1

# Calculate equally weighted portfolio return
equally_weighted_allocation = np.ones(len(tickers)) / len(tickers)
equally_weighted_portfolio_return = np.sum(equally_weighted_allocation * daily_returns, axis=1)
cumulative_return_equally_weighted = np.cumprod(1 + equally_weighted_portfolio_return) - 1

# Calculate random portfolio return
np.random.seed(42)  # For reproducibility
random_allocation = np.random.rand(len(tickers))
random_allocation /= np.sum(random_allocation)  # Normalize weights to sum up to 1
random_portfolio_return = np.sum(random_allocation * daily_returns, axis=1)
cumulative_return_random = np.cumprod(1 + random_portfolio_return) - 1

# Plot cumulative returns of all portfolios
plt.figure(figsize=(10, 6))
plt.plot(cumulative_return_portfolio.index, cumulative_return_portfolio, label='Optimized Portfolio', color='green')
plt.plot(cumulative_return_equally_weighted.index, cumulative_return_equally_weighted, label='Equally Weighted Portfolio', color='blue')
plt.plot(cumulative_return_random.index, cumulative_return_random, label='Random Portfolio', color='red')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Portfolio Cumulative Return Comparison')
plt.legend()
plt.grid(True)
plt.show()