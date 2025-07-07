import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller



# Read and create df
aapl = pd.read_csv("aapl_daily.csv", index_col="Date", parse_dates=["Date"])
aapl = aapl.rename(columns={"Adj Close": "AdjClose"})

# Return df
aapl_lr = np.log(aapl["AdjClose"]).diff().dropna()

print(f'ADF pvalue: {adfuller(aapl_lr)}')

