import pandas as pd
import numpy as np

# 180 days data generate karenge
dates = pd.date_range(start="2024-01-01", periods=180)

data = []

stock = 1000

for date in dates:
    # Weekend demand zyada
    if date.weekday() >= 5:
        units_sold = np.random.randint(45, 65)
    else:
        units_sold = np.random.randint(25, 45)

    stock -= units_sold

    # Agar stock kam ho jaye to refill simulate karo
    if stock < 200:
        stock += 800

    data.append([date.strftime("%Y-%m-%d"), "Milk", units_sold, stock])

df = pd.DataFrame(data, columns=["date", "product", "units_sold", "current_stock"])

df.to_csv("sample_data.csv", index=False)

print("Large dataset generated successfully!")