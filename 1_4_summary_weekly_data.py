import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


# Renaming columns and set Date as Index
fx_prices = pd.read_csv("exchange_rates_1.csv", index_col="Date", parse_dates=["Date"])
fx_prices.columns = ["USDINR", "GBPINR", "EURINR"]

# Resampling data to a weekly data-set
fx_weekly_prices = fx_prices.resample("W").first()
# Let's plot the time series
fig1,ax1 = plt.subplots(figsize=(12,8))
_ = fx_weekly_prices.plot(ax=ax1)
_ = plt.ylabel("Fx Rates")
_ = plt.title("Fx Rates of Indian Rupees vs USD, GBP and EUR", fontsize=16)
sns.despine()
plt.tight_layout()

# Return dataframe
fx_weekly_returns = fx_weekly_prices.apply(lambda x: np.log(x) - np.log(x).shift(1)).dropna()
# Let's plot returns
fig2,ax2 = plt.subplots(figsize=(12,8))
_ = fx_weekly_returns.plot(ax=ax2)
_ = plt.ylabel("Return")
_ = plt.title("Returns of FX Rates of USDINR, GBPINR and EURINR pairs", fontsize=16)
sns.despine()
plt.tight_layout()

# Compute stats
weekly_stats = pd.concat({"obs": fx_weekly_returns.apply(lambda x: x.shape[0]),
                   "mean": fx_weekly_returns.mean(axis=0),
                   "std": fx_weekly_returns.std(axis=0),
                   "autocorr": fx_weekly_returns.apply(lambda s: s.autocorr(lag=1)),
                   "ac_bounds": fx_weekly_returns.apply(lambda s: f"=+/-{np.round(2/np.sqrt(s.shape[0]), 4)}")
                    }, axis=1)
print(weekly_stats)

# T-test
weekly_t_test_p_value = pd.DataFrame({"Weekly T-test p-value": fx_weekly_returns.apply(lambda x: ttest_1samp(x,0)[1])})
print(weekly_t_test_p_value)
# Ljung-Box test
weekly_ljungbox_test_p_value = pd.DataFrame({"Weekly LJungbox p-value": fx_weekly_returns.apply(lambda x: acorr_ljungbox(x, lags=1)["lb_pvalue"].iloc[0])})
print(weekly_ljungbox_test_p_value)
# Jarque-Bera test
weekly_jarquebera_test_p_value = pd.DataFrame({'Weekly Jarque-Bera p-value': fx_weekly_returns.apply(lambda x: jarque_bera(x)[1])})
print(weekly_jarquebera_test_p_value)

# Let's plot the distribution
fig3,ax3 = plt.subplots(2, 2, figsize=(16, 10))
_ = fx_weekly_returns.hist(bins=30, ax=ax3.reshape(4)[0:3])
_ = ax3[1,1].set_visible(False)
plt.show()
