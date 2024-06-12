# Portfolio Optimization App

This project is a web application designed to optimize a stock portfolio using deep learning techniques. It provides users with the ability to input stock tickers, specify a date range, and set various parameters to optimize their portfolio allocation. The app also supports dynamic rebalancing of the portfolio.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Optimize portfolio allocation using deep learning
- Dynamic rebalancing of the portfolio
- Visualization of cumulative returns for optimized, equally weighted, and random portfolios
- Radar chart visualization of the current allocation

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/portfolio_app.git
    cd portfolio_app
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your database configuration:**

    Ensure you have a `config.yaml` file with your database credentials:

    ```yaml
    database:
      host: your_db_host
      database: your_db_name
      user: your_db_user
      password: your_db_password
    ```

5. **Run the application:**

    ```bash
    python app.py
    ```

6. **Access the application:**

    Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. **Navigate to the main page:**

    Here, you can input the stock tickers, start date, end date, rebalancing interval, regularization parameter, and diversification penalty.

2. **Optimize Portfolio:**

    Click on the "Optimize Portfolio" button to perform the optimization.

3. **View Results:**

    The results page will display the optimized portfolio allocation, rebalanced allocations over time, and various plots visualizing cumulative returns and current allocations.

## Project Structure

portfolio_app/
│
├── app.py # Flask application entry point
├── config.yaml # Database configuration file
├── Data_Preparation.py # Data preparation script
├── Deep_Learning.py # Deep learning model and training script
├── Rebalancing.py # Portfolio rebalancing script
├── static/
│ ├── portfolio_cumulative_returns.png # Cumulative returns plot
│ └── current_allocation.png # Current allocation radar chart
├── templates/
│ ├── index.html # Main page HTML
│ └── result.html # Results page HTML
├── README.md # Project README file
├── requirements.txt # Project dependencies
└── pycache/ # Compiled Python files


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any changes you'd like to make.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

