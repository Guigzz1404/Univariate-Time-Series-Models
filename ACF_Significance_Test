import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf


# pandas display
pd.options.display.max_columns = None
pd.options.display.width = None


# If ACF is out the border of the conf interval, then it's significant
def is_lag_significant(acf, c_m, c_u):
    return True if acf > c_u or acf < c_m else False


def color_false_red(val):
    color = 'red' if val == True else 'black'
    return f'color: {color}'


# Formating the df
fx_rates = pd.read_csv("exchange_rates_2.csv", index_col="DATE", parse_dates=["DATE"])
fx_rates = fx_rates.drop(columns=["Unnamed: 0", "days"])
fx_rates.index.name="Date"
fx_rates = fx_rates.dropna()

# Currencies and returns
currencies = fx_rates.columns
fx_returns = fx_rates.apply(lambda x: np.log(x) - np.log(x.shift())).dropna()

acf_info = []

for cur in currencies:
    fxr = fx_returns.loc[:,cur]
    # Return acf value and confidence interval [min,max]
    acf_vals, conf_int = acf(fxr, nlags=10, alpha=0.05)
    df = pd.DataFrame([(fxr.name, i, is_lag_significant(acf_vals[i], conf_int[i][0] - acf_vals[i], conf_int[i][1] - acf_vals[i])) for i in range (1,11)],
                      columns=["Currency", "Lag", "Significant"])
    acf_info.append(df)

acf_info = pd.concat(acf_info).set_index(['Lag','Currency']).unstack()
acf_info.style.map(color_false_red)
print(acf_info)

