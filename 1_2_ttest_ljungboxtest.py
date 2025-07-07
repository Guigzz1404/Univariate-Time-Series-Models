import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox


# Renaming columns and set Date as Index
fx_prices = pd.read_csv("exchange_rates_1.csv", index_col="Date", parse_dates=["Date"])
fx_prices.columns = ["USDINR", "GBPINR", "EURINR"]

# Return df creation
fx_returns = fx_prices.apply(lambda x: np.log(x) - np.log(x).shift(1)).dropna()

# Let's review the main statistics of the returns
stats = pd.concat({"obs": fx_returns.apply(lambda x: x.shape[0]),
                   "mean": fx_returns.mean(axis=0),
                   "std": fx_returns.std(axis=0),
                   "autocorr": fx_returns.apply(lambda s: s.autocorr(lag=1)),
                   "ac_bounds": fx_returns.apply(lambda s: f"=+/-{np.round(2/np.sqrt(s.shape[0]),4)}")
                   }, axis=1)
print(stats)

# Let's test if the mean is significantly different from 0 with one sample t-test analysis
# If proba_value <0.05, the mean is significantly different from 0
t_test_p_value = pd.DataFrame({"T-test p-value": fx_returns.apply(lambda x: ttest_1samp(x,0)[1])})
print(t_test_p_value)

# Let's test if the first order autocorrelation is significantly different from 0 with the ljungbox test
# If proba_value <0.05, autocorrelation depend on time and so the serie is not stationnary (ACF lag1!=0)
ljungbox_test_p_value = pd.DataFrame({'LJung-Box p-value': fx_returns.apply(lambda x: acorr_ljungbox(x, lags=1)['lb_pvalue'].iloc[0])})
print(ljungbox_test_p_value)

