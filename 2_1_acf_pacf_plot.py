import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
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

# Plot of USDINR return, ACF and PACF
fig1,ax1 = plt.subplots(figsize=(12,8))
_ = plt.plot(fx_returns.USDINR)
_ = ax1.set_title("Return series of USDINR")
fig2,(ax2,ax3) = plt.subplots(2,1,figsize=(16,10))
_ = plot_acf(fx_returns.USDINR, ax=ax2, zero=True,  alpha=0.05, lags=10)
_ = ax2.set_title('ACF for USDINR',fontsize=16)
_ = plot_pacf(fx_returns.USDINR, ax=ax3, zero=True, alpha=0.05, lags=10)
_ = ax3.set_title('PACF for USDINR',fontsize=16)
sns.despine()
plt.tight_layout()

# Plot of GBPINR return, ACF and PACF
fig3,ax4 = plt.subplots(figsize=(12,8))
_ = plt.plot(fx_returns.GBPINR)
_ = ax4.set_title("Return series of GBPINR")
fig4,(ax5,ax6) = plt.subplots(2, 1, figsize=(16,10))
_ = plot_acf(fx_returns.GBPINR, ax=ax5, zero=True, alpha=0.05, lags=10)
_ = ax5.set_title("ACF for GBPINR")
_ = plot_pacf(fx_returns.GBPINR, ax=ax6, zero=True, alpha=0.05, lags=10)
_ = ax6.set_title("PACF for GBPINR")
sns.despine()
plt.tight_layout()

# Plot of EURINR return, ACF and PACF
fig5,ax7 = plt.subplots(figsize=(12,8))
_ = plt.plot(fx_returns.EURINR)
_ = ax7.set_title("Return series of EURINR")
fig6,(ax8,ax9) = plt.subplots(2, 1, figsize=(16,10))
_ = plot_acf(fx_returns.EURINR, ax=ax8, zero=True, alpha=0.05, lags=10)
_ = ax8.set_title("ACF for EURINR")
_ = plot_pacf(fx_returns.EURINR, ax=ax9, zero=True, alpha=0.05, lags=10)
_ = ax9.set_title("PACF for EURINR")
sns.despine()
plt.tight_layout()

plt.show()



