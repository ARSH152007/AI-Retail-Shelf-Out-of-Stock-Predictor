import numpy as np

def calculate_reorder_point(avg_daily_demand, lead_time, safety_stock):
    """
    Reorder Point Formula:
    (Average Daily Demand × Lead Time) + Safety Stock
    """
    return (avg_daily_demand * lead_time) + safety_stock


def calculate_stockout_days(current_stock, avg_daily_demand):
    """
    Estimate how many days stock will last
    """
    if avg_daily_demand == 0:
        return np.inf
    return current_stock / avg_daily_demand


def risk_level(days_left):
    """
    Classify risk based on days left
    """
    if days_left <= 3:
        return "HIGH RISK"
    elif days_left <= 7:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"