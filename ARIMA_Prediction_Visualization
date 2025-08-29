import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA


# sns parameters
sns.set_theme(style="darkgrid")

# Read and create df
aapl = pd.read_csv("aapl_daily.csv",index_col="Date",parse_dates=True).rename(columns={"Adj Close":"AdjClose"})

log_volume = np.log(aapl["Volume"])

logv_train = log_volume[:-60] # df to train the ARIMA model
logv_val = log_volume[-60:] # df to test the model with data it never sees

# Plot the series, its ACF and PACF to see if there is any correlation
fig1,(ax1,ax2,ax3) = plt.subplots(3,1, figsize=(12,15))
_ = logv_train.plot(ax=ax1, title="Log-volume series for AAPL")
_ = plot_acf(logv_train, ax=ax2, zero=False, alpha=0.05, lags=30, title="ACF for log Volume of AAPL")
_ = plot_pacf(logv_train, ax=ax3, zero=False, alpha=0.05, lags=30, title="PACF for Log Volume of AAPL")
sns.despine()
plt.tight_layout()

# As the serie seems correlated let's look at the differenced data
logv_diff = logv_train.diff().dropna()
fig2,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,15))
_ = logv_diff.plot(ax=ax1,title='Diff Log-volume series for AAPL')
_ = plot_acf(logv_diff, ax=ax2,zero = True,alpha=0.05,lags=30,title='ACF for Diff Log Volume of AAPL')
_ = plot_pacf(logv_diff, ax=ax3, zero = False,alpha=0.05,lags=30,title='PACF for Diff Log Volume of AAPL')
sns.despine()
plt.tight_layout()

# ACF significant at lag 1 and 2 and PACF significant over several lags. Let's try ARIMA(3,1,2)
model1 = ARIMA(logv_train, order=(3,1,2))
results1 = model1.fit()
print(results1.summary())
# L3 coeff no significant. Let's find a more parsimonus model
model2 = ARIMA(logv_train, order=(1,1,1))
results2 = model2.fit()
print(results2.summary()) # AIC & BIC smaller so it's better

# Plot residuals
residuals = pd.DataFrame(results2.resid[1:])
fig3, ax = plt.subplots(1,2,figsize=(12,8))
_ = residuals.plot(title="Residuals", ax=ax[0],legend=False)
_ = residuals.plot(kind='kde', title='Density', ax=ax[1],legend=False)
plt.tight_layout()
sns.despine()

# Get prediction
pred = results2.get_prediction(dynamic=False)
pred_mean = pd.Series(pred.predicted_mean.values, index=logv_train.index)
# Prediction of futur mean
pred_futur = results2.get_forecast(steps=60)
pred_futur_mean = pd.Series(pred_futur.predicted_mean.values, index=logv_val.index)

# Plot the comparison
fig4, ax = plt.subplots(figsize=(12,8))
_ = pred_mean[1:].plot(ax=ax, label="Prediction", legend=True)
_ = logv_train.plot(ax=ax, legend=True)
sns.despine()
plt.tight_layout()
plt.show()
