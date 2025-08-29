import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


# Read and create df
aapl = pd.read_csv("aapl_daily.csv", index_col="Date", parse_dates=["Date"]).rename(columns={"Adj Close": "AdjClose"})
# Create price series
pt = aapl["AdjClose"].apply(np.log)

model = ARIMA(pt, order=(0,0,1))
results = model.fit()
print(results.summary())
