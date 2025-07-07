import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import  r2_score
from scipy.stats import chi2


# Formating the df
fx_rates = pd.read_csv("exchange_rates_2.csv", index_col="DATE", parse_dates=["DATE"])
fx_rates = fx_rates.drop(columns=["Unnamed: 0", "days"])
fx_rates.index.name="Date"
fx_rates = fx_rates.dropna()

# Currencies and returns
currencies = fx_rates.columns
fx_returns = fx_rates.apply(lambda x: np.log(x) - np.log(x.shift())).dropna()

arch_effects = []

for cur in currencies:

    fxr2 = fx_returns.loc[:, cur] ** 2 # Squared return
    arch_lags = pd.concat([fxr2.shift(i) for i in range(1, 6)], axis=1).dropna() # df of shifted lags (ex: 1000rows x 5col)
    fxr2 = fxr2.loc[arch_lags.index] # Reindex serie frx2 to match exactly with date index of arch_lags
    regr = linear_model.LinearRegression()
    _ = regr.fit(arch_lags, fxr2)
    pred = regr.predict(arch_lags)
    r2 = r2_score(fxr2, pred)
    p_value = chi2.pdf(r2 * arch_lags.shape[0], 5)

    arch_effects.append({'Currency': fxr2.name, 'r2': np.round(r2, 3), 'p_value': np.round(p_value, 3)})

arch_effects = pd.DataFrame(arch_effects).set_index('Currency')
print(arch_effects)



