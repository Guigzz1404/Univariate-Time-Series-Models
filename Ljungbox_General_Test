import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
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


# pandas display
pd.options.display.max_columns = None
pd.options.display.width = None

# print(pd.read_csv("aapl_daily.csv").head())
aapl = pd.read_csv("aapl_daily.csv", index_col="Date", parse_dates=["Date"])
aapl = aapl.rename(columns={"Adj Close": "AdjClose"})
# print(aapl.info())
# print(aapl.describe())

# Return df
aapl_lr = np.log(aapl["AdjClose"]).diff().dropna()

# Plot return, ACF and PACF
fig1,ax1 = plt.subplots(figsize=(12,8))
_ = aapl_lr.plot(ax=ax1, title="Log-return series for AAPL")
fig2,(ax2, ax3) = plt.subplots(2, 1, figsize=(12, 15))
_ = plot_acf(aapl_lr, ax=ax2, lags=30, alpha=0.05, zero=True, title="ACF for AAPL")
_ = plot_pacf(aapl_lr, ax=ax3, lags=30, alpha=0.05, zero=False, title="PACF for AAPL")
max_pacf = max(abs(ax3.lines[1].get_ydata()))
ax3.set_ylim(-max_pacf, max_pacf)
sns.despine()
plt.tight_layout()

# LJung-Box to confirm if ACF is significant and so if there is a serial correlation
print(acorr_ljungbox(aapl_lr, lags=1, return_df=True))

plt.show()




