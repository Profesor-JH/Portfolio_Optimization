import torch
from torch.utils.data import DataLoader, TensorDataset
import mysql.connector
import pandas as pd
import yaml

# Read the configuration file
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Access database credentials from the configuration

host = config['database']['host']
database = config['database']['database']
user = config['database']['user']
password = config['database']['password']

def prepare_data(tickers, start_date, end_date):
    """
    Function to download stock data and prepare the dataset from our server
    """
    # Establish a connection to the MySQL database
    connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
    cursor = connection.cursor()

    # Initialize an empty DataFrame to store the data
    data = pd.DataFrame()

    # Fetch data for each ticker
    for ticker in tickers:
        # Fetch data from the database
        query = f"SELECT Date, Close FROM Trading_Data WHERE Ticker = '{ticker}' AND Date BETWEEN '{start_date}' AND '{end_date}'"
        cursor.execute(query)
        ticker_data = cursor.fetchall()

        # Convert fetched data to DataFrame
        ticker_df = pd.DataFrame(ticker_data, columns=['Date', ticker])
        ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
        ticker_df.set_index('Date', inplace=True)

        # Append data for current ticker to the overall DataFrame
        data = pd.concat([data, ticker_df], axis=1)

    # Close the database connection
    cursor.close()
    connection.close()

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    # Prepare inputs and targets for the model
    inputs = torch.tensor(daily_returns.values[:-1], dtype=torch.float32)  # Using returns as inputs
    targets = torch.tensor(daily_returns.values[1:], dtype=torch.float32)  # Predicting next day's returns

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return data_loader, daily_returns


