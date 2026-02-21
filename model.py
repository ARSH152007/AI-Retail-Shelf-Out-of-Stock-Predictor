import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def prepare_features(data):
    # Safe datetime conversion
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Invalid dates become NaT
    
    # Drop rows with invalid/missing dates
    data = data.dropna(subset=['date'])
    
    # Time-based features
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['weekday'] = data['date'].dt.weekday
    
    # Rolling average (last 7 days demand)
    data['rolling_avg_7'] = data['units_sold'].rolling(window=7).mean()
    
    # Drop rows with NaN from rolling avg
    data = data.dropna()
    
    return data


def train_model(data):
    data = prepare_features(data)
    
    features = ['day', 'month', 'weekday', 'rolling_avg_7']
    X = data[features]
    y = data['units_sold']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, data


def forecast_next_days(model, data, days=7):
    last_date = pd.to_datetime(data['date'].iloc[-1])
    predictions = []
    
    rolling_avg = data['units_sold'].rolling(window=7).mean().iloc[-1]
    
    for i in range(1, days + 1):
        future_date = last_date + pd.Timedelta(days=i)
        
        features = pd.DataFrame({
            'day': [future_date.day],
            'month': [future_date.month],
            'weekday': [future_date.weekday()],
            'rolling_avg_7': [rolling_avg]
        })
        
        pred = model.predict(features)[0]
        predictions.append(pred)
        
        # Update rolling average for next iteration
        rolling_avg = (rolling_avg * 6 + pred) / 7
    
    return predictions