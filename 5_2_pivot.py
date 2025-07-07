import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox


# Read and create df
aapl = pd.read_csv("aapl_daily.csv", index_col="Date", parse_dates=["Date"])
aapl = aapl.rename(columns={"Adj Close": "AdjClose"})

# Pivot creation
pivot_1 = (aapl.High + aapl.Low)/2
pivot_2 = (aapl.High + aapl.Low + aapl.Close)/3
p1_lr = np.log(pivot_1).diff().dropna()
p2_lr = np.log(pivot_2).diff().dropna()

# LJung Box test
print(acorr_ljungbox(p1_lr, lags=1, return_df=True))
print(acorr_ljungbox(p2_lr, lags=1, return_df=True))

