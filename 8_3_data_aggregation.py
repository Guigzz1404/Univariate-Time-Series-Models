import pandas as pd
import numpy as np
import datetime
import order_book_creation


# Get num trades, vwap, and average duration each 2min
def aggregate_trades_1(df):
    num_trades = df.Quantity.shape[0]
    if df.Quantity.sum() != 0:
        vwap = (df.Quantity * df.Price).sum() / df.Quantity.sum() # vwap calculation
    else:
        vwap = np.nan
    dur_ms = df.index.diff().mean().microseconds / 1000 # average duration
    return pd.Series({'N': num_trades, 'VWAP': vwap, 'DurationMs': dur_ms})


# Candle creation
def aggregate_trades_2(df):
    volume = df.Quantity.sum()
    if volume == 0:
        p_open = np.NaN
        p_close = np.NaN
        p_hi = np.NaN
        p_lo = np.NaN
    else:
        p_open = df.Price.iloc[0]
        p_close = df.Price.iloc[-1]
        p_hi = df.Price.max()
        p_lo = df.Price.min()

    return pd.Series({'Open': p_open, 'Close': p_close, 'High': p_hi, 'Low': p_lo, 'Volume': volume})


# Creation of df
biof_data = pd.read_csv("biof_levelIII_data.csv")
origin = datetime.date.today()
biof_data["Timestamp"] = pd.to_datetime(biof_data["Timestamp"],unit='ms',origin=origin)
biof_data = biof_data.drop(columns=['Ticker','MPID'])

order_book,trades = order_book_creation.process_book_data(biof_data, datetime.datetime.combine(origin,  datetime.time(11,0,0)))
trades = trades.set_index("Timestamp")

agg_1 = trades.groupby(pd.Grouper(freq="2min")).apply(aggregate_trades_1)
print(agg_1)
agg_2 = trades.groupby(pd.Grouper(freq="2min")).apply(aggregate_trades_2)
print(agg_2)
