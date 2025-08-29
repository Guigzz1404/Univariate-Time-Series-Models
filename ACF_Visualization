import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns


# sns parameters
sns.set_theme(style="darkgrid")

# Read and create df
aapl = pd.read_csv("aapl_daily.csv", index_col="Date", parse_dates=["Date"]).rename(columns={"Adj Close": "AdjClose"})
# Create price series
pt = aapl["AdjClose"].apply(np.log)

# Plot pt
fig, ax = plt.subplots(1, 2, figsize=(18,8))
_ = pt.plot(ax=ax[0])
_ = ax[0].set_ylabel("p_t$")
_ = ax[0].set_title("Apple log-price data", fontsize=14)
_ = plot_acf(pt, lags=70, ax=ax[1])
_ = ax[1].set_title("Autocorrelation of p_t$", fontsize=14)
sns.despine()
plt.tight_layout()
plt.show()




