# Portfolio Optimization App

This project is a web application designed to optimize a stock portfolio using deep learning techniques. It provides users with the ability to input stock tickers, specify a date range, and set various parameters to optimize their portfolio allocation. The app also supports dynamic rebalancing of the portfolio.

**Watch a demo here:** https://www.youtube.com/watch?v=7zYVZbyyTAw&t=2s


# Table of Contents

- [Technologies](#Technologies Used:)
- [Features](#Key Features:)
- [Responsibilities](#Responsibilities)
- [Impact](#Impact)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)


## Technologies Used:
**Frontend:**

HTML/CSS: For creating the user interface templates.
**Backend:**

* Flask: For handling web server functionality and routing.

* Python: For data processing and implementing business logic.
Data Handling:

* Pandas: For data manipulation and preparation.
SQLAlchemy: For MySQL RDS (AWS) database interactions.
Machine Learning:

* TensorFlow/Keras: For developing and training deep learning models.
Visualization:

* Matplotlib: For generating plots and charts.
Configuration:

* YAML: For managing database configuration securely.

## Key Features:

**Portfolio Optimization:**

* Deep Learning Integration: Implemented a deep learning model for predicting stock prices and optimizing portfolio allocation.
Rebalancing Script: Automated portfolio rebalancing based on the latest market data and user-defined criteria.
Dynamic Data Handling:

* Data Preparation: Scripted data preprocessing steps to clean and normalize financial data.
Visualization: Created plots for cumulative returns and radar charts for current allocation to help users visualize their portfolio performance.
User Interface:

* Interactive Web Pages: Designed HTML templates for displaying the main page and results dynamically.
User Inputs: Enabled users to input their preferences and constraints for portfolio optimization.

## Responsibilities:

**Backend Development:**

* Developed Flask routes and integrated the web application with backend logic.
Designed and managed the database schema for storing financial data and user preferences.
Data Processing and Machine Learning:

* Wrote scripts for data preparation, ensuring clean and normalized input for models.
Built and trained deep learning models for stock price prediction using TensorFlow/Keras.
User Interface and Visualization:

* Created and styled HTML templates for a user-friendly interface.
Implemented data visualization using Matplotlib to provide insights into portfolio performance.

**Automation and Deployment:**

Automated the portfolio rebalancing process to reflect the latest data.
Deployed the Flask application to a web server for public access.

## Impact:

**Enhanced Decision Making:**

Provided users with an advanced tool to optimize their investment portfolios using cutting-edge machine learning techniques.
Improved user experience through dynamic visualizations and intuitive web interface.

**Operational Efficiency:**

Automated data preparation and portfolio rebalancing, significantly reducing manual effort.
Enabled real-time portfolio analysis and optimization based on the latest market data.

**Scalability:**

Designed the application to be easily extensible, allowing for the addition of new features and financial metrics in the future.


## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Profesor-JH/Portfolio_Optimization.git
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

```

Portfolio_Optimization/
├── app.py                          # Flask application entry point
├── config.yaml                     # Database configuration file
├── Data_Preparation.py             # Data preparation script
├── Deep_Learning.py                # Deep learning model and training script
├── Rebalancing.py                  # Portfolio rebalancing script
├── static/
│   ├── portfolio_cumulative_returns.png  # Cumulative returns plot
│   └── current_allocation.png      # Current allocation radar chart
├── templates/
│   ├── index.html                  # Main page HTML
│   └── result.html                 # Results page HTML
├── README.md                       # Project README file
├── requirements.txt                # Project dependencies
└── __pycache__/                    # Compiled Python files

```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any changes you'd like to make.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

=======
Check demo here : https://youtu.be/7zYVZbyyTAw?si=Ja-vzYfUh5oLgLZ8

