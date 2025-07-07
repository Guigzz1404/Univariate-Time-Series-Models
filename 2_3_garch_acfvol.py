import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import warnings
from statsmodels.graphics.tsaplots import plot_acf

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

# GARCH model for USDINR
res_usd = arch_model(fx_returns.USDINR, mean="Constant", p=1, q=1).fit()
print(res_usd.summary())
res_gbp = arch_model(fx_returns.GBPINR, mean="Constant", p=1, q=1).fit()
print(res_gbp.summary())
res_eur = arch_model(fx_returns.EURINR, mean="Constant", p=1, q=1).fit()
print(res_eur)


# Plot
# USDINR
fig1, (ax1,ax2) = plt.subplots(2,1,figsize=(12,15))
_ = (np.sqrt(250)*res_usd.conditional_volatility).plot(ax=ax1)
_ = ax1.set_title('Annualized Conditional Volatility for USDINR')
_ = ax1.set_ylabel('Conditional Vol')
_ = plot_acf(res_usd.conditional_volatility,ax=ax2)
_ = ax2.set_title('ACF of Conditional Volatility')
_ = ax2.set_xlabel('Lags')
_ = ax2.set_ylabel('ACF')
sns.despine()
plt.tight_layout()

# GBPINR
fig2, (ax3,ax4) = plt.subplots(2, 1, figsize=(12,15))
_ = (np.sqrt(250)*res_gbp.conditional_volatility).plot(ax=ax3)
_ = ax3.set_title("Annualized Conditional Volatility for GBPINR")
_ = ax3.set_ylabel("Conditional Vol")
_ = plot_acf(res_gbp.conditional_volatility, ax=ax4)
_ = ax4.set_title("ACF of Conditional Volatility")
_ = ax4.set_xlabel("Lags")
_ = ax4.set_ylabel("ACF")
sns.despine()
plt.tight_layout()

# EURINR
fig3, (ax5,ax6) = plt.subplots(2, 1, figsize=(12,15))
_ = (np.sqrt(250)*res_eur.conditional_volatility).plot(ax=ax5)
_ = ax5.set_title("Annualized Conditional Volatility for EURINR")
_ = ax5.set_ylabel("Conditional Vol")
_ = plot_acf(res_eur.conditional_volatility, ax=ax6)
_ = ax6.set_title("ACF of Conditional Volatility")
_ = ax6.set_xlabel("Lags")
_ = ax6.set_ylabel("ACF")
sns.despine()
plt.tight_layout()

# Concat the parameters of 3 time series
print(pd.concat((res_usd.params,res_gbp.params,res_eur.params),axis=1,keys=['USDINR','GBPINR','EURINR']).T)

plt.show()
