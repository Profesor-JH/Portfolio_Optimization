# risk_manager.py
import numpy as np
import pandas as pd


def calculate_var(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    return cvar

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0):
    """
    Calculate Sharpe Ratio for the portfolio.
    """
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio

def calculate_sortino_ratio(daily_returns, risk_free_rate=0):
    """
    Calculate Sortino Ratio for the portfolio.
    """
    mean_return = daily_returns.mean()
    downside_std = daily_returns[daily_returns < 0].std()
    sortino_ratio = (mean_return - risk_free_rate) / downside_std
    return sortino_ratio
