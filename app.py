import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import train_model, forecast_next_days
from utils import calculate_reorder_point, calculate_stockout_days, risk_level

# Streamlit page config
st.set_page_config(page_title="AI Retail Stock Predictor", layout="wide")
st.title("🛒 AI Retail Shelf Out-of-Stock Predictor")
st.write("Upload your retail sales dataset to predict demand and prevent stock-outs.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Check for required columns
    required_cols = ['date', 'units_sold', 'current_stock']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"CSV missing required column(s): {', '.join(missing_cols)}")
        st.stop()

    # Safe preview
    st.subheader("Raw Data Preview")
    st.dataframe(data.head())

    # Train Model
    try:
        model, processed_data = train_model(data)
        st.success("Model trained successfully!")
    except Exception as e:
        st.error(f"Error training model: {e}")
        st.stop()

    # Forecast
    forecast_days = st.slider("Select number of days to forecast", 1, 30, 7)
    try:
        predictions = forecast_next_days(model, processed_data, forecast_days)
    except Exception as e:
        st.error(f"Error during forecasting: {e}")
        st.stop()

    avg_demand = sum(predictions) / len(predictions)
    current_stock = data["current_stock"].iloc[-1]

    # Inventory Inputs
    st.sidebar.header("Inventory Settings")
    lead_time = st.sidebar.number_input("Supplier Lead Time (days)", value=3)
    safety_stock = st.sidebar.number_input("Safety Stock", value=100)

    reorder_point = calculate_reorder_point(avg_demand, lead_time, safety_stock)
    days_left = calculate_stockout_days(current_stock, avg_demand)
    risk = risk_level(days_left)

    # Forecast Output
    st.subheader("📈 Forecasted Demand")
    st.write(predictions)

    # Inventory Insights
    st.subheader("📦 Inventory Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Forecasted Demand", f"{avg_demand:.2f}")
    col2.metric("Current Stock", current_stock)
    col3.metric("Days Until Stockout", f"{days_left:.2f}")

    st.write(f"### 🔔 Risk Level: {risk}")
    st.write(f"### 📌 Reorder Point: {reorder_point:.2f}")

    if current_stock <= reorder_point:
        st.error("⚠ Reorder Immediately!")
    else:
        st.success("Stock Level Safe")

    # Plot Forecast
    st.subheader("📊 Forecast Graph")
    fig, ax = plt.subplots()
    ax.plot(range(1, forecast_days + 1), predictions, marker='o', color='blue')
    ax.set_xlabel("Future Days")
    ax.set_ylabel("Predicted Units Sold")
    ax.set_title("Forecasted Units Sold for Next Days")
    st.pyplot(fig)