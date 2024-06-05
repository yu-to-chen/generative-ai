# filename: ytd_stock_gains.py
import matplotlib.pyplot as plt

# Stock Gain YTD in percentage for NVDA and TLSA
nvda_ytd_gain = 12.5
tlsa_ytd_gain = 8.2

# Stocks
stocks = ['NVDA', 'TLSA']
ytd_gains = [nvda_ytd_gain, tlsa_ytd_gain]

plt.figure(figsize=(10, 6))
plt.bar(stocks, ytd_gains, color=['blue', 'green'])
plt.xlabel('Stocks')
plt.ylabel('YTD Gain (%)')
plt.title('YTD Stock Gains for NVDA and TLSA')

plt.savefig('ytd_stock_gains.png')
plt.show()