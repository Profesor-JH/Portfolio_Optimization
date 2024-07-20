from flask import Flask, render_template, request
import torch
from Data_Preparation import prepare_data
from Deep_Learning import Portfolio, loss_function, train_model
from Rebalancing import rebalance_portfolio
import numpy as np
from itertools import zip_longest
from risk_manager import calculate_var, calculate_cvar, calculate_sharpe_ratio, calculate_sortino_ratio
import cvxpy as cp

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_1.html')

@app.route('/optimize_portfolio', methods=['POST'])
def optimize_portfolio():
    # Retrieve user inputs from the HTML form
    tickers = request.form['tickers'].split(',')
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    rebalance_interval = int(request.form['rebalance_interval'])
    lambda_reg = float(request.form['lambda_reg'])
    diversification_reg = float(request.form['diversification_reg'])
    
    # Call the prepare_data function to get the data loader and daily returns
    data_loader, daily_returns = prepare_data(tickers, start_date, end_date)
    
    # Set hyperparameters
    input_size = len(tickers)
    output_size = len(tickers)
    learning_rate = 0.001
    
    # Initialize model, optimizer, and data loader
    model = Portfolio(input_size=input_size, output_size=output_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, data_loader, optimizer)
    
    # Get the asset allocation from the optimized portfolio
    with torch.no_grad():
        allocation = model(data_loader.dataset.tensors[0]).numpy()[-1]

    # Perform dynamic rebalancing of the portfolio
    rebalanced_allocations, rebalance_dates = rebalance_portfolio(model, daily_returns, rebalance_interval)
    rebalanced_allocations_enum = list(enumerate(rebalanced_allocations))

    # Calculate portfolio returns for optimized, equally weighted, and mean-variance optimized portfolios
    portfolio_daily_return = np.sum(allocation * daily_returns, axis=1)
    cumulative_return_portfolio = np.cumprod(1 + portfolio_daily_return) - 1

    equally_weighted_allocation = np.ones(len(tickers)) / len(tickers)
    equally_weighted_portfolio_return = np.sum(equally_weighted_allocation * daily_returns, axis=1)
    cumulative_return_equally_weighted = np.cumprod(1 + equally_weighted_portfolio_return) - 1

    # Mean-Variance Optimization
    returns = np.mean(daily_returns, axis=0).values.reshape(-1, 1)  # Convert to NumPy array and ensure it's a column vector

    cov_matrix = np.cov(daily_returns.T)  # Ensure covariance matrix is correctly shaped

    # Debugging prints
    print(f"Returns (shape {returns.shape}): {returns}")
    print(f"Daily returns (shape {daily_returns.shape}): {daily_returns}")
    print(f"Covariance matrix (shape {cov_matrix.shape}): {cov_matrix}")

    # Variables
    weights = cp.Variable(len(tickers))
    gamma = cp.Parameter(nonneg=True, value=lambda_reg)  # Use lambda_reg as gamma
    div_reg = cp.Parameter(nonneg=True, value=diversification_reg)  # Diversification regularization

    expected_return = returns.T @ weights  # Matrix multiplication
    risk = cp.quad_form(weights, cov_matrix)
    diversification_penalty = cp.norm(weights, 2)  # L2 regularization term

    # Objective: maximize return - risk
    objective = cp.Maximize(expected_return - gamma * risk - div_reg * diversification_penalty)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    mv_optimized_allocation = weights.value
    mv_portfolio_return = np.sum(mv_optimized_allocation * daily_returns, axis=1)
    cumulative_return_mv_optimized = np.cumprod(1 + mv_portfolio_return) - 1

    # Calculate risk metrics
    var = {
        'optimized': calculate_var(portfolio_daily_return),
        'equally_weighted': calculate_var(equally_weighted_portfolio_return),
        'mean_variance_optimized': calculate_var(mv_portfolio_return)
    }

    cvar = {
        'optimized': calculate_cvar(portfolio_daily_return),
        'equally_weighted': calculate_cvar(equally_weighted_portfolio_return),
        'mean_variance_optimized': calculate_cvar(mv_portfolio_return)
    }

    sharpe_ratio = {
        'optimized': calculate_sharpe_ratio(portfolio_daily_return),
        'equally_weighted': calculate_sharpe_ratio(equally_weighted_portfolio_return),
        'mean_variance_optimized': calculate_sharpe_ratio(mv_portfolio_return)
    }

    sortino_ratio = {
        'optimized': calculate_sortino_ratio(portfolio_daily_return),
        'equally_weighted': calculate_sortino_ratio(equally_weighted_portfolio_return),
        'mean_variance_optimized': calculate_sortino_ratio(mv_portfolio_return)
    }

    # Prepare data to pass to the template
    context = {
        'tickers': tickers,
        'allocation': allocation.tolist(),
        'equally_weighted_allocation': equally_weighted_allocation.tolist(),
        'mean_variance_optimized_allocation': mv_optimized_allocation.tolist(),
        'rebalanced_allocations_enum': rebalanced_allocations_enum,
        'rebalance_dates': rebalance_dates,
        'cumulative_return_portfolio': cumulative_return_portfolio[-1],
        'cumulative_return_equally_weighted': cumulative_return_equally_weighted[-1],
        'cumulative_return_mean_variance_optimized': cumulative_return_mv_optimized[-1],
        'var': var,
        'cvar': cvar,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio
    }
    
    return render_template('index_1.html', **context, zip=zip_longest)

if __name__ == '__main__':
    app.run(debug=True)
