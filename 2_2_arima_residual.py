import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns


# Renaming columns and set Date as Index
fx_prices = pd.read_csv("exchange_rates_1.csv", index_col="Date", parse_dates=["Date"])
fx_prices.columns = ["USDINR", "GBPINR", "EURINR"]

# Return df creation
fx_returns = fx_prices.apply(lambda x: np.log(x) - np.log(x).shift(1)).dropna()

# MA(3) model because ACF dependency on lag3
results = ARIMA(fx_returns.GBPINR, order=(0, 0, 3)).fit()
print(results.summary())

# Plot residual (observed value - model (predicted) value)
fig,ax = plt.subplots(figsize=(12,8))
_ = plt.plot(results.resid)
_ = plt.ylabel('Residuals')
_ = plt.xlabel('Date')
_ = ax.set_title('MA(3) Residual Plot')
sns.despine()
plt.tight_layout()
plt.show()