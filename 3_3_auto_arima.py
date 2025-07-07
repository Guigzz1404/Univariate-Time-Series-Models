import pandas as pd
import numpy as np
import pmdarima as pm
import warnings

# Hide warnings which are no big issues (put "default" to see warnings)
warnings.simplefilter("ignore")


# pandas display
pd.options.display.max_columns = None
pd.options.display.width = None

# Formating the df
fx_rates = pd.read_csv("exchange_rates_2.csv", index_col="DATE", parse_dates=["DATE"])
fx_rates = fx_rates.drop(columns=["Unnamed: 0", "days"])
fx_rates.index.name="Date"
fx_rates = fx_rates.dropna()

# Currencies and returns
currencies = fx_rates.columns
fx_returns = fx_rates.apply(lambda x: np.log(x) - np.log(x.shift())).dropna()

arma_best = []

for cur in currencies:

    fxr = fx_returns.loc[:,cur]
    model = pm.auto_arima(fxr, start_p=1, start_q=1,max_p=3, max_q=3, error_action='ignore', suppress_warnings=True, trace=False)
    arma = {'Currency':cur, 'AR_Order':model.order[0],'MA_Order':model.order[2]}
    arma_best.append(arma)
arma_best = pd.DataFrame(arma_best).set_index('Currency')
print(arma_best[(arma_best.AR_Order + arma_best.MA_Order) > 0])
