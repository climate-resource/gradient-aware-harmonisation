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
# # Gradient aware harmonisation (Getting Started)
#
# Here we introduce a method for harmonising two timeseries.
# This part may be more unusual or unfamiliar
# to people used to working with arrays,
# so it serves as an introduction
# into some of the concepts used in this package.

# %% [markdown]
# ## Imports

# %%
# Imports
from functools import partial

import numpy as np

from gradient_aware_harmonisation import harmonise
from gradient_aware_harmonisation.convergence import (
    get_polynomial_decay_harmonised_spline,
)
from gradient_aware_harmonisation.plotting import plotting
from gradient_aware_harmonisation.timeseries import Timeseries

# %% [markdown]
# ## Toy Example 1: Artificial data
# ### Create some data

# %%
# create some data
harmonisation_time = 3.0

x1 = np.arange(-2, 3.0, 0.1)
y1 = -16 * x1

x2 = np.arange(2, 10, 0.1)
y2 = 0.5 * x2 + x2**3

target_timeseries = Timeseries(time_axis=x1, values=y1)
harmonisee_timeseries = Timeseries(time_axis=x2, values=y2)

# %% [markdown]
# ### Harmonise functions using `harmonise`
# #### Inspect `harmoniser` function

# %%
help(harmonise)

harmonised_timeseries = harmonise(
    target_timeseries=target_timeseries,
    harmonisee_timeseries=harmonisee_timeseries,
    harmonisation_time=harmonisation_time,
    convergence_time=None,
)

# %% [markdown]
# ### Plot results

# %%
plotting(
    harmonisee_timeseries,
    target_timeseries,
    harmonised_timeseries,
    harmonisation_time,
    convergence_time=None,
)

# %% [markdown]
# ### Introduce a `convergence_time` at $x=8.$

# %%
convergence_time = 8.0

harmonised_timeseries = harmonise(
    target_timeseries=target_timeseries,
    harmonisee_timeseries=harmonisee_timeseries,
    harmonisation_time=harmonisation_time,
    convergence_time=convergence_time,
)

plotting(
    harmonisee_timeseries,
    target_timeseries,
    harmonised_timeseries,
    harmonisation_time,
    convergence_time,
)

# %% [markdown]
# ### Use polynomial-decay

# %%
convergence_time = 8.0

harmonised_timeseries = harmonise(
    target_timeseries=target_timeseries,
    harmonisee_timeseries=harmonisee_timeseries,
    harmonisation_time=harmonisation_time,
    convergence_time=convergence_time,
    get_harmonised_spline=partial(get_polynomial_decay_harmonised_spline, power=2.0),
)

plotting(
    harmonisee_timeseries,
    target_timeseries,
    harmonised_timeseries,
    harmonisation_time,
    convergence_time,
)
# %% [markdown]
# ## Toy Example 2: Use timeseries data
# ### Read data

# %%
# select harmonization time point
harmonisation_time = 2004
convergence_time = 2006

# get timeseries
target_timeseries = Timeseries(
    time_axis=np.array([2001, 2002, 2003, 2004]),
    values=np.array([371.77, 373.72, 376.33, 378.83]),
)
harmonisee_timeseries = Timeseries(
    time_axis=np.array([2003, 2004, 2005, 2006, 2007]),
    values=np.array([375.56, 376.28, 378.83, 381.20, 382.55]),
)

# %% [markdown]
# ### Run `harmonise` for different settings

# %%
# harmonise timeseries at t0
harmonised_timeseries = harmonise(
    target_timeseries=target_timeseries,
    harmonisee_timeseries=harmonisee_timeseries,
    harmonisation_time=harmonisation_time,
    convergence_time=None,
)

# harmonise timeseries at t0 and assure convergence at t1 (converge_t)
harmonised_timeseries2 = harmonise(
    target_timeseries=target_timeseries,
    harmonisee_timeseries=harmonisee_timeseries,
    harmonisation_time=harmonisation_time,
    convergence_time=convergence_time,
)

# %% [markdown]
# ### Plot results

# %%
plotting(
    harmonisee_timeseries=harmonisee_timeseries,
    target_timeseries=target_timeseries,
    interpolated_timeseries=harmonised_timeseries,
    harmonisation_time=harmonisation_time,
    convergence_time=None,
)

plotting(
    harmonisee_timeseries,
    target_timeseries,
    harmonised_timeseries2,
    harmonisation_time,
    convergence_time,
)
