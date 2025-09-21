import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.linear_model import LinearRegression

st.title("Electricity Usage vs. Prices Dashboard")

# Ensure merged.csv exists
if not os.path.exists("data/merged.csv"):
    import preprocessing  # runs preprocessing.py

# Load data
df = pd.read_csv("data/merged.csv", parse_dates=["date"])

# Train models if not present
def train_and_save_models(df):
    X = df["date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    usage_y = df["usage_kwh"].values
    price_y = df["price_per_kwh"].values
    cost_y = df["daily_cost"].values

    usage_model = LinearRegression().fit(X, usage_y)
    price_model = LinearRegression().fit(X, price_y)
    cost_model = LinearRegression().fit(X, cost_y)

    os.makedirs("models", exist_ok=True)
    with open("models/usage_model.pkl", "wb") as f:
        pickle.dump(usage_model, f)
    with open("models/price_model.pkl", "wb") as f:
        pickle.dump(price_model, f)
    with open("models/cost_model.pkl", "wb") as f:
        pickle.dump(cost_model, f)

if not (os.path.exists("models/usage_model.pkl") and os.path.exists("models/price_model.pkl") and os.path.exists("models/cost_model.pkl")):
    train_and_save_models(df)

# Charts for historical data
st.subheader("Electricity Usage (kWh)")
st.line_chart(df.set_index("date")["usage_kwh"])

st.subheader("Electricity Price ($/kWh)")
st.line_chart(df.set_index("date")["price_per_kwh"])

st.subheader("Daily Cost ($)")
st.line_chart(df.set_index("date")["daily_cost"])

# Forecast section
st.header("AI Forecasts (Next 90 Days)")

# Load models
with open("models/usage_model.pkl", "rb") as f:
    usage_model = pickle.load(f)
with open("models/price_model.pkl", "rb") as f:
    price_model = pickle.load(f)
with open("models/cost_model.pkl", "rb") as f:
    cost_model = pickle.load(f)

# Generate future dates
future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=90, freq="D")
X_future = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

forecast_usage = usage_model.predict(X_future)
forecast_price = price_model.predict(X_future)
forecast_cost = cost_model.predict(X_future)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast_usage": forecast_usage,
    "forecast_price": forecast_price,
    "forecast_cost": forecast_cost
})

st.subheader("Forecasted Electricity Usage (kWh)")
st.line_chart(forecast_df.set_index("date")["forecast_usage"])

st.subheader("Forecasted Electricity Price ($/kWh)")
st.line_chart(forecast_df.set_index("date")["forecast_price"])

st.subheader("Forecasted Daily Cost ($)")
st.line_chart(forecast_df.set_index("date")["forecast_cost"])

st.subheader("Data Preview")
st.write(df.tail())

# Seasonality Analysis
st.header("Seasonality Analysis")

# Extract month names
df["month"] = df["date"].dt.strftime("%B")

# Compute average usage per month
monthly_usage = df.groupby("month")["usage_kwh"].mean()

# Reorder months properly (Jan → Dec)
months_order = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]
monthly_usage = monthly_usage.reindex(months_order)

# Bar chart
st.subheader("Average Monthly Electricity Usage (kWh)")
st.bar_chart(monthly_usage)

# Grouped Seasonality Analysis (Usage vs. Cost)
st.subheader("Grouped Seasonality: Usage vs. Cost")

monthly_usage_cost = pd.DataFrame({
    "Usage (kWh)": monthly_usage,
    "Cost ($)": monthly_cost
}, index=months_order)


# Seasonality Analysis Toggle
st.header("Seasonality Analysis")

view_option = st.radio(
    "Choose seasonality chart type:",
    ["Static (Streamlit)", "Interactive (Altair)"]
)

if view_option == "Static (Streamlit)":
    st.subheader("Grouped Seasonality: Usage vs. Cost (Static)")
    st.bar_chart(monthly_usage_cost)
else:
    import altair as alt
    st.subheader("Grouped Seasonality: Usage vs. Cost (Interactive)")

    monthly_usage_cost_reset = monthly_usage_cost.reset_index().melt(
        id_vars="index", var_name="Metric", value_name="Value"
    )
    monthly_usage_cost_reset.rename(columns={"index": "Month"}, inplace=True)

    chart = alt.Chart(monthly_usage_cost_reset).mark_bar().encode(
        x=alt.X("Month:N", sort=months_order, title="Month"),
        y=alt.Y("Value:Q", title="Average"),
        color="Metric:N",
        tooltip=["Month", "Metric", "Value"]
    ).properties(width=700)

    st.altair_chart(chart, use_container_width=True)


import altair as alt

# Interactive grouped bar chart with Altair
st.subheader("Interactive Seasonality: Usage vs. Cost")

monthly_usage_cost_reset = monthly_usage_cost.reset_index().melt(id_vars="index", var_name="Metric", value_name="Value")
monthly_usage_cost_reset.rename(columns={"index": "Month"}, inplace=True)

chart = alt.Chart(monthly_usage_cost_reset).mark_bar().encode(
    x=alt.X("Month:N", sort=months_order, title="Month"),
    y=alt.Y("Value:Q", title="Average"),
    color="Metric:N",
    tooltip=["Month", "Metric", "Value"]
).properties(width=700)

st.altair_chart(chart, use_container_width=True)



# Average monthly cost seasonality
monthly_cost = df.groupby("month")["daily_cost"].mean()
monthly_cost = monthly_cost.reindex(months_order)




# Allow users to download seasonality data
st.subheader("Download Seasonality Data")

seasonality_csv = monthly_usage_cost.reset_index()
seasonality_csv.rename(columns={"index": "Month"}, inplace=True)

st.download_button(
    label="📥 Download Seasonality Data as CSV",
    data=seasonality_csv.to_csv(index=False).encode("utf-8"),
    file_name="seasonality_usage_cost.csv",
    mime="text/csv"
)
