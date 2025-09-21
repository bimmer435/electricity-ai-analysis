import os
import pandas as pd
import numpy as np

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# If usage.csv and prices.csv don't exist, generate synthetic data
if not os.path.exists("data/usage.csv") or not os.path.exists("data/prices.csv"):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=730, freq="D")
    usage = 30 + 10*np.sin(np.linspace(0, 12*np.pi, 730)) + np.random.normal(0, 3, 730)
    usage = np.clip(usage, 15, None)
    price = 0.14 + 0.00003*np.arange(730) + np.random.normal(0, 0.002, 730)
    price = np.clip(price, 0.12, None)

    usage_df = pd.DataFrame({"date": dates, "usage_kwh": usage})
    price_df = pd.DataFrame({"date": dates, "price_per_kwh": price})

    usage_df.to_csv("data/usage.csv", index=False)
    price_df.to_csv("data/prices.csv", index=False)
    print("Synthetic usage.csv and prices.csv generated.")

# Merge into merged.csv
usage = pd.read_csv("data/usage.csv")
prices = pd.read_csv("data/prices.csv")
merged = pd.merge(usage, prices, on="date")
merged["daily_cost"] = merged["usage_kwh"] * merged["price_per_kwh"]
merged.to_csv("data/merged.csv", index=False)
print("Merged dataset saved to data/merged.csv")
