import torch
# Import the prepare_data function from Data_Preparation.py
from Data_Preparation import prepare_data
# Import the modelling code
from Deep_Learning import Portfolio, loss_function, train_model
# Import the rebalancing function
from Rebalancing import rebalance_portfolio  
import torch.optim as optim
# visualization
import matplotlib.pyplot as plt
import numpy as np


# Define your list of tickers and date range
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
start_date = '2023-02-27'
end_date = '2024-02-27'

# Call the prepare_data function to get the data loader and daily returns
data_loader, daily_returns = prepare_data(tickers, start_date, end_date)

# Set hyperparameters
input_size = len(tickers)  # Assuming tickers is the list of stock tickers
output_size = len(tickers)  # Output size is same as input size for portfolio optimization
learning_rate = 0.001
rebalance_interval = 20  # Define rebalancing interval (e.g., monthly)


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


# Perform dynamic rebalancing of the portfolio
rebalanced_allocations, rebalance_dates = rebalance_portfolio(model, daily_returns, rebalance_interval)


# Plot the rebalanced allocations as a bar plot
plt.figure(figsize=(10, 6))
tick_labels = tickers  # Assuming tickers is defined
for i, ticker in enumerate(tickers):
    plt.bar(rebalance_dates, [alloc[i] for alloc in rebalanced_allocations], label=ticker)

plt.xlabel('Date')
plt.ylabel('Allocation')
plt.title('Rebalanced Portfolio Allocations Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Calculate the portfolio returns with dynamic rebalancing
portfolio_returns_rebalanced = []
for i, alloc in enumerate(rebalanced_allocations):
    portfolio_return = np.sum(alloc * daily_returns.iloc[i * rebalance_interval])
    portfolio_returns_rebalanced.append(portfolio_return)

# Check the data type of portfolio_returns_rebalanced
print(type(portfolio_returns_rebalanced))

# Check the structure and data type of each element in portfolio_returns_rebalanced
for item in portfolio_returns_rebalanced:
    print(type(item))

# Check the length of portfolio_returns_rebalanced
print(len(portfolio_returns_rebalanced))

# Calculate the cumulative return of the rebalanced portfolio
cumulative_return_rebalanced = np.cumprod(1 + portfolio_returns_rebalanced) - 1
print(cumulative_return_rebalanced)

# Plot the cumulative return of the rebalanced portfolio
plt.figure(figsize=(10, 6))
plt.plot(rebalance_dates, cumulative_return_rebalanced, label='Rebalanced Portfolio', color='green')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Rebalanced Portfolio Cumulative Return Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
