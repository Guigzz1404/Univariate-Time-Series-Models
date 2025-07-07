import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Plot first 5 lines to check content
# print(pd.read_csv("exchange_rates_1.csv").head())

# Renaming columns and set Date as Index
fx_prices = pd.read_csv("exchange_rates_1.csv", index_col="Date", parse_dates=["Date"])
fx_prices.columns = ["USDINR", "GBPINR", "EURINR"]
# print(fx_prices.round(4).head())

# Let's look at the data to ensure all is sensible
fx_prices.info()
print('\n')
# No nulls and reasonable numbers we can move forward
# describe function permits to plot the mean, st. dev., min, first quartile, median, third quartile and max of each column
print(fx_prices.describe())

# Let's plot the time series
fig1,ax1 = plt.subplots(figsize=(12,8))
# This line permits to create the graph with index of df as x-axis and columns of df with values of the graph
_ = fx_prices.plot(ax=ax1)
_ = plt.ylabel('Fx Rates')
_ = plt.title('Fx Rates of Indian Rupees vs USD, GBP and EUR',fontsize=16)
# Remove the frame of the graph
sns.despine()
# When we have subgraph use this to avoid superimposition
plt.tight_layout()

# Let's generate the returns and have a quick look. We'll favor log return
fx_returns = fx_prices.apply(lambda x: np.log(x) - np.log(x).shift(1)).dropna()
fig2,ax2 = plt.subplots(figsize=(12,8))
_ = fx_returns.plot(ax=ax2)
_ = plt.ylabel("Return")
_ = plt.title("Returns of FX Rates of USDINR, GBPINR and EURINR pairs",fontsize=16)
sns.despine()
plt.tight_layout()
plt.show()
