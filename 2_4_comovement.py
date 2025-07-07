import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import warnings

# Plot design
sns.set_theme(style="darkgrid")
register_matplotlib_converters()
warnings.filterwarnings('ignore')
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Renaming columns and set Date as Index
fx_prices = pd.read_csv("exchange_rates_1.csv", index_col="Date", parse_dates=["Date"])
fx_prices.columns = ["USDINR", "GBPINR", "EURINR"]

# Return df creation
fx_returns = fx_prices.apply(lambda x: np.log(x) - np.log(x).shift(1)).dropna()

# Plot series with same starting point
fig,ax = plt.subplots(figsize=(12,8))
_ = (1+fx_returns).cumprod().plot(ax=ax)
_ = plt.title("Normalized Price Plot")
plt.tight_layout()
sns.despine()
plt.show()
