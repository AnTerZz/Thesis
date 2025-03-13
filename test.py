import eikon as ek
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Point, Daily

# Set your Refinitiv API Key
ek.set_app_key('95ce4c898667464ba3f23364d5c91adfc9811a60')

# Fetch Apple's daily closing stock price for 2023
stock_df = ek.get_timeseries(
    "AAPL.O", 
    fields="CLOSE", 
    start_date="2023-01-01", 
    end_date="2023-12-31", 
    interval="daily"
)
stock_df = stock_df.rename(columns={'CLOSE': 'AAPL_Close'})
stock_df.index = pd.to_datetime(stock_df.index)  # Ensure date format is consistent

# Location: Vancouver, BC
vancouver = Point(49.2497, -123.1193, 70)

# Fetch weather data for Vancouver (full-year 2023)
weather_2023 = Daily(vancouver, datetime(2023, 1, 1), datetime(2023, 12, 31)).fetch()
weather_2023 = weather_2023[['tavg']].rename(columns={'tavg': 'AvgTemp2023'})
weather_2023.index = pd.to_datetime(weather_2023.index)  # Ensure date format matches

# Fetch historical average temperature (climatology: 2013-2022)
climatology = Daily(vancouver, datetime(2013, 1, 1), datetime(2022, 12, 31)).fetch()
climatology = climatology.groupby([climatology.index.month, climatology.index.day]).tavg.mean()
climatology.index.names = ['month', 'day']

# Calculate temperature deviations (2023 - historical average)
weather_2023['Month'] = weather_2023.index.month
weather_2023['Day'] = weather_2023.index.day
weather_2023 = weather_2023.reset_index()

# Join climatology
weather_2023['HistoricalAvg'] = weather_2023.apply(
    lambda row: climatology.loc[(row['Month'], row['Day'])], axis=1
)
weather_2023['TempDeviation'] = weather_2023['AvgTemp2023'] - weather_2023['HistoricalAvg']

weather_2023.set_index('time', inplace=True)

# **Filter weather data to include only trading days**
weather_2023 = weather_2023.loc[stock_df.index]  # Keeps only dates that exist in stock data

# Merge datasets on the common trading days
combined_df = stock_df.join(weather_2023[['TempDeviation']], how='inner')

# Drop any NaN values (in case of missing weather data on some trading days)
combined_df.dropna(inplace=True)

# Plotting stock prices and temperature deviations
fig, ax1 = plt.subplots(figsize=(12,6))

# Apple stock price
color_stock = 'tab:blue'
ax1.set_xlabel('Date (2023)')
ax1.set_ylabel('AAPL Stock Price (USD)', color=color_stock)
ax1.plot(combined_df.index, combined_df['AAPL_Close'], color=color_stock, label='AAPL Close')
ax1.tick_params(axis='y', labelcolor=color_stock)

# Temperature deviation on second axis
ax2 = ax1.twinx()
color_temp = 'tab:red'
ax2.set_ylabel('Temp Deviation (°C)', color=color_temp)
ax2.plot(combined_df.index, combined_df['TempDeviation'], color=color_temp, linestyle='--', label='Temp Deviation')
ax2.tick_params(axis='y', labelcolor=color_temp)

# Formatting & legend
fig.tight_layout()
fig.autofmt_xdate()
plt.title('Apple Stock Price and Temperature Deviations (Vancouver) in 2023')

# Create a unified legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.grid(alpha=0.3)
plt.show()
