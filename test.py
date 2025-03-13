import eikon as ek
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from meteostat import Point, Daily
from dotenv import load_dotenv
import numpy as np
import os

# Load API keys from environment variables
load_dotenv()
ek.set_app_key(os.getenv("REFINITIV_API_KEY"))

### 1. Fetch Air France stock data ###
stock_df = ek.get_timeseries(
    "AIRF.PA", 
    fields="CLOSE", 
    start_date="2010-01-01", 
    end_date="2024-12-31", 
    interval="daily"
)
stock_df = stock_df.rename(columns={'CLOSE': 'Stock_Close'})
stock_df.index = pd.to_datetime(stock_df.index)  # Ensure datetime format

### 2. Fetch Weather Data for Paris ###
paris = Point(48.8566, 2.3522, 35)  # Paris coordinates

# Fetch 2023 temperature data
weather_2023 = Daily(paris, datetime(2010, 1, 1), datetime(2024, 12, 31)).fetch()
weather_2023 = weather_2023[['prcp']].rename(columns={'prcp': 'Rainfall'})
weather_2023.index = pd.to_datetime(weather_2023.index)

# Fetch historical climatology data (2013-2022)
climatology = Daily(paris, datetime(2013, 1, 1), datetime(2022, 12, 31)).fetch()
climatology = climatology.groupby([climatology.index.month, climatology.index.day]).tavg.mean()
climatology.index.names = ['month', 'day']

# Calculate temperature deviations
weather_2023['Month'] = weather_2023.index.month
weather_2023['Day'] = weather_2023.index.day
weather_2023 = weather_2023.reset_index()

# Add historical average temperature
weather_2023['HistoricalAvg'] = weather_2023.apply(
    lambda row: climatology.loc[(row['Month'], row['Day'])], axis=1
)
weather_2023['Deviation'] = weather_2023['Rainfall'] - weather_2023['HistoricalAvg']
weather_2023.set_index('time', inplace=True)

# **Filter weather data to include only trading days**
weather_2023 = weather_2023.loc[stock_df.index]  

# Merge datasets on the common trading days
combined_df = stock_df.join(weather_2023[['Deviation']], how='inner')

# Drop missing values
combined_df.dropna(inplace=True)

# Calculate daily log returns
combined_df['Stock_Return'] = combined_df['Stock_Close'].pct_change().apply(lambda x: np.log(1 + x))
# Filter extreme weather days
rain_threshold = weather_2023['Rainfall'].quantile(0.95)
extreme_weather_df = combined_df[abs(combined_df['Deviation']) > rain_threshold].dropna()

# Define independent (X) and dependent (Y) variables
X = extreme_weather_df[['Deviation']]
X = sm.add_constant(X)  # Add constant
Y = extreme_weather_df['Stock_Return']

# Run OLS regression
model = sm.OLS(Y, X).fit()

# Print regression summary
print(model.summary())

### 5. Plot Results ###
fig, ax1 = plt.subplots(figsize=(12,6))

# Stock price
ax1.set_xlabel('Date (2023)')
ax1.set_ylabel('Air France Stock Price (EUR)', color='tab:blue')
ax1.plot(combined_df.index, combined_df['Stock_Close'], color='tab:blue', label='Air France Close')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Temperature deviation on second axis
ax2 = ax1.twinx()
ax2.set_ylabel('Deviation (°C)', color='tab:red')
ax2.plot(combined_df.index, combined_df['Deviation'], color='tab:red', linestyle='--', label='Deviation')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Formatting & legend
fig.tight_layout()
fig.autofmt_xdate()
plt.title('Air France Stock Price and Extreme Weather Deviations in Paris (2023)')

# Create a unified legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.grid(alpha=0.3)
plt.show()


### 6. Volatility Analysis ##

# Split dataset into extreme and normal weather days
extreme_days = combined_df[abs(combined_df['Deviation']) > 5]['Stock_Return'].dropna()
normal_days = combined_df[abs(combined_df['Deviation']) <= 5]['Stock_Return'].dropna()


vol_extreme = extreme_days.std()
vol_normal = normal_days.std()

print(f"Volatility on extreme weather days: {vol_extreme:.4f}")
print(f"Volatility on normal days: {vol_normal:.4f}")

if vol_extreme > vol_normal:
    print("Stock is more volatile on extreme weather days.")
else:
    print("Stock is not significantly more volatile on extreme weather days.")
