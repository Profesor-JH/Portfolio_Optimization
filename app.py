import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

from flask import Flask, render_template, request
import torch
from Data_Preparation import prepare_data
from Deep_Learning import Portfolio, loss_function, train_model
from Rebalancing import rebalance_portfolio
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
import seaborn as sns
import matplotlib as mpl
import matplotlib.dates as mdates

app = Flask(__name__)

# Define your Flask routes
@app.route('/')
def index():
    return render_template('index.html')

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

    # Perform dynamic rebalancing of the portfolio
    rebalanced_allocations, rebalance_dates = rebalance_portfolio(model, daily_returns, rebalance_interval)
    # Enumerate rebalanced_allocations
    rebalanced_allocations_enum = list(enumerate(rebalanced_allocations))

    # Calculate portfolio returns for optimized, equally weighted, and random portfolios
    portfolio_daily_return = np.sum(allocation * daily_returns, axis=1)
    cumulative_return_portfolio = np.cumprod(1 + portfolio_daily_return) - 1

    equally_weighted_allocation = np.ones(len(tickers)) / len(tickers)
    equally_weighted_portfolio_return = np.sum(equally_weighted_allocation * daily_returns, axis=1)
    cumulative_return_equally_weighted = np.cumprod(1 + equally_weighted_portfolio_return) - 1

    np.random.seed(42)  # For reproducibility
    random_allocation = np.random.rand(len(tickers))
    random_allocation /= np.sum(random_allocation)  # Normalize weights to sum up to 1
    random_portfolio_return = np.sum(random_allocation * daily_returns, axis=1)
    cumulative_return_random = np.cumprod(1 + random_portfolio_return) - 1

    # Set Seaborn style to "darkgrid"
    # Set Seaborn style to "darkgrid" with grey grid and smaller grid lines
    sns.set_style("darkgrid", {"axes.grid": True, "grid.color": "grey", "grid.linewidth": 0.1})


    # Plot cumulative returns with Seaborn
    plt.figure(figsize=(7, 6))  # Adjust figure size for better readability

    # Plotting cumulative returns for optimized, equally weighted, and random portfolios
    sns.lineplot(data=cumulative_return_portfolio, label='Optimized Portfolio - Deep Learning', color='#4CAF50')
    sns.lineplot(data=cumulative_return_equally_weighted, label='Equally Weighted Portfolio', color='#2196F3')
    sns.lineplot(data=cumulative_return_random, label='Random Portfolio', color='#FFD700')

    # Set labels and title with white text color
    plt.xlabel('Date', fontsize=12, color='white')  # Adjust font size and color for better readability
    plt.ylabel('Cumulative Return', fontsize=12, color='white')
    plt.title('', fontsize=14, color='white')  # Adjust font size and color for better readability

    # Format the dates on the x-axis
    date_form = mdates.DateFormatter("%Y-%m-%d")  # Define the date format
    plt.gca().xaxis.set_major_formatter(date_form)  # Set the date format for the x-axis

    # Set x and y axis labels color to white
    plt.gca().xaxis.label.set_color('white')
    plt.gca().yaxis.label.set_color('white')

        # Set x and y axis tick labels color to white
    plt.gca().tick_params(axis='x', colors='white')
    plt.gca().tick_params(axis='y', colors='white')

    # Add legend and adjust layout
    plt.legend(fontsize=10)  # Adjust font size for better readability
    plt.tight_layout()  # Adjust layout to prevent overlapping elements

    # Save the plot as an image with transparent background
    plt.savefig('static/portfolio_cumulative_returns.png', transparent=True)

    # Close the plot to free up memory
    plt.close()

    # Set Seaborn style to "darkgrid"
    sns.set_style("darkgrid")

    # Set background color to black
    #mpl.rcParams['axes.facecolor'] = 'black'

    # Allocate data
    allocation = rebalanced_allocations_enum[-1][1]

    # Plot radar chart with Seaborn
    plt.figure(figsize=(6, 6))

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Make background transparent
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # Compute evenly spaced angles for radar chart
    theta = np.linspace(0, 2*np.pi, len(tickers), endpoint=False).tolist()
    theta += theta[:1]  # To connect the plot

    # Plot the allocation values
    allocation_r = np.concatenate((allocation, [allocation[0]]))  # To connect the plot
    ax.fill(theta, allocation_r, 'b', alpha=0.1)  # Fill the area

    # Plot the allocation values as points
    ax.plot(theta, allocation_r, linestyle='-', marker='o', color='white')  # Set point color to white

    # Annotate each point with its allocation value
    for i, (angle, alloc) in enumerate(zip(theta, allocation)):
        # Calculate text position slightly outside the plot lines
        text_x = angle
        text_y = alloc + 0.04  # Adjust the y-coordinate as needed

        # Add text with white color and larger font size
        ax.text(text_x, text_y, f'{alloc:.2f}', ha='center', va='bottom', fontsize=12, color='white')

    # Add labels
    ax.set_ylim(0, 0.3)  # Adjust the limit for better visualization
    # Set x ticks and labels with white color
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(tickers, color='white', fontsize=14)

    # Adjust grid size and color
    ax.grid(color='grey', linewidth=0.5)  # Set grid color to grey and linewidth to 0.5


    # Show the radar plot
    plt.title('')
    plt.tight_layout()
    plt.savefig('static/current_allocation.png')  # Save the plot as an image
    plt.close()

    print(cumulative_return_portfolio)
    # Prepare data to pass to the template
    context = {
        'tickers': tickers,
        'allocation': allocation,
        'rebalanced_allocations_enum': rebalanced_allocations_enum,
        'rebalance_dates': rebalance_dates,
        'cumulative_return_portfolio': cumulative_return_portfolio
    }
    
    return render_template('result.html', **context, zip=zip_longest)

if __name__ == '__main__':
    app.run(debug=True)
