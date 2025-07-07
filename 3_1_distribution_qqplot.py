import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import warnings
from statsmodels.graphics.gofplots import qqplot

# pandas display
pd.options.display.max_columns = None
pd.options.display.width = None
# plots
sns.set_theme(style="darkgrid")
register_matplotlib_converters()
warnings.filterwarnings('ignore')
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# print(pd.read_csv("exchange_rates_2.csv").head())

# Formating the df
fx_rates = pd.read_csv("exchange_rates_2.csv", index_col="DATE", parse_dates=["DATE"])
fx_rates = fx_rates.drop(columns=["Unnamed: 0", "days"])
fx_rates.index.name="Date"
fx_rates = fx_rates.dropna()

# print(fx_rates.head())
# print(fx_rates.info())
# print(fx_rates.describe())

# Currencies and returns
currencies = fx_rates.columns
fx_returns = fx_rates.apply(lambda x: np.log(x) - np.log(x.shift())).dropna()

for cur in currencies:

    fxr = fx_returns.loc[:, cur]
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
    _ = fig.suptitle(f"Histogram and QQ-Plot for {cur}", fontsize=20)
    _ = fxr.plot.hist(ax=ax1, bins=20, density=True)
    _ = ax1.set_xlabel("Returns")
    qqplot(fxr, ax=ax2, line="45", fit=True)
    fig.subplots_adjust(top=0.91)
    plt.tight_layout()
    sns.despine()
    fig.show()

plt.show()