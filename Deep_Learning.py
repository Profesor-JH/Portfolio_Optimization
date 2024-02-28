# Import the prepare_data function from Data_Preparation.py
from Data_Preparation import prepare_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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