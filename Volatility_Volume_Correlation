import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# sns parameters
sns.set_theme(style="darkgrid")


def vol_estimates(df):
    o = np.log(df["Open"]) - np.log(df["Close"].shift())
    u = np.log(df["High"]) - np.log(df["Open"])
    d = np.log(df["Low"]) - np.log(df["Open"])
    c = np.log(df["Close"]) - np.log(df["Open"])

    n = c.shape[0]
    k = 0.34/(1.34 + (n+1)*(n-1))

    v_cc = ((o+c)-(o.mean()+c.mean())).pow(2).sum()/(n-1) # Vol Close to Close
    v_rs = (u*(u-c)+d*(d-c)).sum()/n # Vol Rogers & Satchell
    v_yz = (o-o.mean()).pow(2).sum()/(n-1) + k*(c-c.mean()).pow(2).sum()/(n-1) + (1-k)*v_rs # Vol Yang & Zhang

    return pd.Series({"Vcc":v_cc, "Vrs":v_rs, "Vyz": v_yz})


# Read and create df
aapl = pd.read_csv("aapl_daily.csv", index_col="Date", parse_dates=["Date"])
aapl = aapl.rename(columns={"Adj Close": "AdjClose"})

# Group df every 22 days and compute volatility
vol_measures = aapl.groupby(pd.Grouper(freq="22d")).apply(vol_estimates)
# Compute every 22 days the mean log volume
log_volume = aapl.Volume.groupby(pd.Grouper(freq="22d")).apply(lambda x: np.log(x).mean())

# Plot
fig, ax = plt.subplots(figsize=(12,8))
_ = vol_measures.iloc[:,0:3].plot(ax=ax, linestyle="-", marker="o")
_ = log_volume.plot(ax=ax,secondary_y=True,label=None,legend=True,marker='o')
sns.despine()
plt.tight_layout()
plt.show()

# Correlation between volatility and log volume
corr = vol_measures.corrwith(log_volume)
print(corr)

