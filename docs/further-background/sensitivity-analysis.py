# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gradient aware harmonisation (sensitivity-analyses)
#
# Here we introduce a method for harmonising two timeseries.
# This part may be more unusual or unfamiliar
# to people used to working with arrays,
# so it serves as an introduction
# into some of the concepts used in this package.

# %%
# ## Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from gradient_aware_harmonisation.harmonise import harmonise

# %% [markdown]
# ## Sensitivity of results wrt number of observations
# ### Setup simulation study

# %%
# initialize result tables
## for absolute value
df_res = dict(
    N = [],
    historical = [],
    abs = [],
    abs_slope = [],
    interpolated = []
)
## for first-order derivatives
df_res2 = deepcopy(df_res)

# set-up simulations
t0 = 3.0                                    # harmonisation time
num_obs = [10,30,50,100,300,1_000, 5_000]   # number of data points

# simulate results across number of obs.
for i in num_obs:
    x1 = np.linspace(-2, 4., i)
    y1 = -6*x1

    x2 = np.linspace(2, 20, i)
    y2 = 0.5*x2 + x2**3

    res = harmonise((x1,x2), (y1,x2), t0)

    for df, f in zip([df_res, df_res2], ["f","df"]):
        df["N"].append(i)
        df["historical"].append(res[f"{f}1"][-1])
        df["abs"].append(res[f"{f}2_abs"][0])
        df["abs_slope"].append(res[f"{f}2_adj"][0])
        df["interpolated"].append(res[f"{f}2_intpol"][0])

# %% [markdown]
# ### Results
# **zero-order derivative**

# %%
# absolute values (zero-order derivative)
pd.DataFrame(df_res).round(2)

# %% [markdown]
# **first-order derivative**

# %%
# first-order derivative
pd.DataFrame(df_res2).round(2)

# %% [markdown]
# ### Plot results

# %%
plt.figure(figsize=(6,3))
plt.plot(res["x2"], res["f2"], label="original", linestyle="--", color="black")
plt.plot(res["x2"], res["f2_intpol"], label="interpolated")
plt.plot(res["x2"], res["f2_abs"], label="abs")
plt.plot(res["x2"], res["f2_adj"], label="abs+slope")
plt.plot(res["x1"], res["f1"], label="historical", color="red")
plt.axvline(t0, color="black", linestyle="dotted")
plt.legend(handlelength=0.3, fontsize="small", frameon=False, loc="lower left")
plt.show()
