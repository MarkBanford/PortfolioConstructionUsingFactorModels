import pandas as pd
import edhec_risk_kit_201 as erk
import warnings
import statsmodels.api as sm
import numpy as np

warnings.filterwarnings('ignore')

brka_d = pd.read_csv("brka_d_ret.csv", parse_dates=True, index_col=0)  # daily returns

# convert to monthly

brka_m = brka_d.resample('M').apply(erk.compound).to_period('M')
print(brka_m.head())

# load Fama-French explanatory variables, monthly factor returns (Mkt-RFR, SMB(Size), HML(Value-Growth), RF)
fff = erk.get_fff_returns()
print(fff.head())

# Decompose the portfolio returns into portion due to market and the rest not market, using CAPM

# R(P)-RF = Alpha + Beta(R(M)-RF) + Error

brka_excess = brka_m["1990": "2012-05"] - fff.loc["1990": "2012-05", ['RF']].values
mkt_excess = fff.loc["1990": "2012-05", ['Mkt-RF']]
exp_var = mkt_excess.copy()
exp_var["Constant"] = 1
lm = sm.OLS(brka_excess, exp_var).fit()

print(lm.summary())  # notice constant in output, alpha is 61bps per month, Beta = 0.5402
