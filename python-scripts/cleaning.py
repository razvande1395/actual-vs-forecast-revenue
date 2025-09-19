import pandas as pd


df = pd.read_csv("retail_sales_dataset.csv")


df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


df = df.dropna(subset=["Date"])


df["Computed_Total"] = df["Quantity"] * df["Price per Unit"]
df["Total Amount"] = df["Computed_Total"]


df = df.drop_duplicates()


df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()


#creating a monthly revenue data set to use later vs forecasting
monthly_sales = (
    df.groupby("Month")["Total Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Total Amount": "Revenue"})
)


df.to_csv("retail_sales_clean.csv", index = False)
monthly_sales.to_csv("retail_sales_monthly.csv", index = False)


print("Cleaning complete. Files saved:")
print("retail_sales_clean.csv (row-level cleaned data)")
print("retail_sales_monthly.csv (monthly revenue)")
