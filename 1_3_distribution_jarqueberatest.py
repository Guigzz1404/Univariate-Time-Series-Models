import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import jarque_bera



# Renaming columns and set Date as Index
fx_prices = pd.read_csv("exchange_rates_1.csv", index_col="Date", parse_dates=["Date"])
fx_prices.columns = ["USDINR", "GBPINR", "EURINR"]

# Return df creation
fx_returns = fx_prices.apply(lambda x: np.log(x) - np.log(x).shift(1)).dropna()

# Plot histogram to see the distribution of the return
fig,ax = plt.subplots(2,2,figsize=(16,10))
_ = fx_returns.hist(bins=30,ax=ax.reshape(4)[0:3])
_ = ax[1,1].set_visible(False)

# Let's test if the distribution is normal with Jarque-Bera test
# If p-value <0.05, we reject the normality of the serie
jarquebera_test_p_value = pd.DataFrame({"Jarque-Bera p-value": fx_returns.apply(lambda x: jarque_bera(x)[1])})
print(jarquebera_test_p_value)
plt.show()


