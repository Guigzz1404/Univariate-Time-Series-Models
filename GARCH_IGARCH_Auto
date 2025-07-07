import pandas as pd
import numpy as np
from arch import arch_model


# Return a dictionary of garch model results
def collect_results_garch(cur, m_id, model):
    out = {}
    out['Currency'] = cur
    out['ID'] = m_id
    out['Converged'] = model.convergence_flag == 0
    out['AIC'] = np.round(model.aic)
    params = model.params.round(4)
    t_values = model.tvalues.round(4)
    out['omega'] = f"{params['omega']} ({t_values['omega']})"
    out['alpha'] = f"{params['alpha[1]']} ({t_values['alpha[1]']})"
    out['beta'] = f"{params['beta[1]']} ({t_values['beta[1]']})"

    return out


# Return a dictionary of garch model results
def collect_results_igarch(cur, m_id, model):
    out = {}
    out['Currency'] = cur
    out['ID'] = m_id
    out['Converged'] = model.convergence_flag == 0
    out['AIC'] = np.round(model.aic)
    params = model.params.round(4)
    t_values = model.tvalues.round(4)
    out['omega'] = f"{params['omega']} ({t_values['omega']})"
    out['phi'] = f"{params['phi']} ({t_values['phi']})"
    out['d'] = f"{params['d']} ({t_values['d']})"
    out['beta'] = f"{params['beta']} ({t_values['beta']})"

    return out


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

garch_models = []

for cur in currencies:

    fxr = fx_returns.loc[:, cur]
    # For normal distribution
    model_N = arch_model(fxr, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=True).fit(disp="off")
    garch_models.append(collect_results_garch(cur, "GARCH11_N", model_N))
    # For t-distribution
    model_T = arch_model(fxr, mean="Constant", vol="GARCH", p=1, q=1, dist="studentst", rescale=True).fit(disp="off")
    garch_models.append(collect_results_garch(cur, "GARCH11_T", model_T))

garch_models = pd.DataFrame(garch_models).set_index(["Currency", "ID"])
print(garch_models)

igarch_models = []

for cur in currencies:
    fxr = fx_returns.loc[:, cur]
    model_IN = arch_model(fxr, mean='Constant', vol='FIGARCH', p=1, q=1, dist='normal', rescale=True).fit(disp='off')
    igarch_models.append(collect_results_igarch(cur, 'GARCH11_IN', model_IN))
    model_IT = arch_model(fxr, mean='Constant', vol='FIGARCH', p=1, q=1, dist='studentst', rescale=True).fit(disp='off')
    igarch_models.append(collect_results_igarch(cur, 'GARCH11_IT', model_IT))

igarch_models = pd.DataFrame(igarch_models).set_index(['Currency', 'ID'])
print(igarch_models)
