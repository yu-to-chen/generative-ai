# filename: get_stock_prices_plot.py
from functions import get_stock_prices, plot_stock_prices
import pandas as pd

# Define the stock symbols and the current date
stock_symbols = ['NVDA', 'TSLA']
current_date = '2024-06-04'
start_date = current_date[:4] + '-01-01'  # YTD start date is the first day of the current year

# Get the stock prices YTD for NVDA and TSLA
stock_prices = get_stock_prices(stock_symbols, start_date, current_date)

# Create a plot for the stock prices YTD and save it to a file
plot_stock_prices(stock_prices, 'stock_prices_YTD_plot.png')