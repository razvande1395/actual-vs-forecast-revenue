import pandas as pd
from prophet import Prophet

df = pd.read_csv("retail_sales_monthly.csv")  # expected columns are Month, Revenue
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month")


df_p = df.rename(columns={"Month": "ds", "Revenue": "y"})


m = Prophet()
m.fit(df_p)

# this forecasts the next n months - depending on the periods parameter
future = m.make_future_dataframe(periods = 6, freq = "M")
fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]


out = fcst.merge(df_p, on = "ds", how = "left")
out["is_actual"] = out["y"].notna()


out.to_csv("sales_forecast.csv", index = False)

print("Saved: sales_forecast.csv")
print("Columns:", list(out.columns))
