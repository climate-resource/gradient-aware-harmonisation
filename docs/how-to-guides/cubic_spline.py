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
# # How to use a cubic-spline as harmonisation of two functions?
# In this tutorial, we present use cases for applying a cubic-spline
# to harmonise two functions which we will call in the following
# `diverge_from` and `harmonisee`.
# The `cubic-spline` interpolates between `diverge_from` and `harmonisee`.


# %%
# import relevant libraries
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from gradient_aware_harmonisation.add_cubic import (
    harmonise_splines_add_cubic,
)
from gradient_aware_harmonisation.spline import SplineScipy

# %% [markdown]

# We start by defining the spline `diverge_from` as a linear
# function with intercept=1.0 and slope=2.5.

# %%
diverge_from_gradient = 2.5
diverge_from_y_intercept = 1.0

diverge_from = SplineScipy(
    scipy.interpolate.PPoly(
        c=[[diverge_from_gradient], [diverge_from_y_intercept]],
        x=[0, 1e8],
    )
)

# %% [markdown]
# ## Scenarios
# ### Harmonisation time < convergence time
# In the following, we consider nine scenarios in which the
# `harmonisee` spline differs from the `diverge_from` spline
# due to varying shifts in the intercept ([0.0, -1.2, 1.2])
# and slope ([1.0, 0.7, 1.4]).
# In all of these scenarios we consider harmonisation time
# (=0) < convergence time (=3.2).

# %%
harmonisation_time = 0.0
convergence_time = 3.2


# %%
def plot_spline(spline, x, ax, label, gradient=False):  # noqa: D103
    ax.plot(
        x,
        spline(x),
        label=label,
    )

    if gradient:
        ax.set_title("Gradient")
    else:
        ax.set_title("Function")


# %%
i = 0
for y_intercept_shift in [0.0, -1.2, 1.2]:
    for gradient_factor in [1.0, 0.7, 1.4]:
        harmonisee = SplineScipy(
            scipy.interpolate.PPoly(
                c=[
                    [diverge_from_gradient * gradient_factor],
                    [diverge_from_y_intercept + y_intercept_shift],
                ],
                x=[0, 1e8],
            )
        )

        res = harmonise_splines_add_cubic(
            diverge_from=diverge_from,
            harmonisee=harmonisee,
            harmonisation_time=harmonisation_time,
            convergence_time=convergence_time,
        )

        fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

        plot_spline(
            diverge_from, np.linspace(-1.0, 3.0, 101), ax=axes[0], label="diverge_from"
        )
        plot_spline(
            harmonisee,
            np.linspace(harmonisation_time, 2 * convergence_time, 101),
            ax=axes[0],
            label="harmonisee",
        )
        plot_spline(
            res,
            np.linspace(harmonisation_time, 2 * convergence_time, 101),
            ax=axes[0],
            label="res",
        )

        plot_spline(
            diverge_from.derivative(),
            np.linspace(-1.0, 3.0, 101),
            ax=axes[1],
            label="diverge_from",
            gradient=True,
        )
        plot_spline(
            harmonisee.derivative(),
            np.linspace(harmonisation_time, 2 * convergence_time, 101),
            ax=axes[1],
            label="harmonisee_gradien",
            gradient=True,
        )
        plot_spline(
            res.derivative(),
            np.linspace(harmonisation_time, 2 * convergence_time, 101),
            ax=axes[1],
            label="cubic-spline",
            gradient=True,
        )

        for ax in axes:
            ax.axvline(
                harmonisation_time,
                label="harmonisation_time",
                color="gray",
                linestyle=":",
            )
            ax.axvline(
                convergence_time, label="convergence_time", color="gray", linestyle="--"
            )
        for ax in axes[1::2]:
            ax.legend(handlelength=1.1, loc="center right", fontsize="small")

        fig.suptitle(
            f"Scenario {i+1} (intercept shift: {y_intercept_shift},"
            + f" slope factor: {gradient_factor})"
        )
        plt.show()
        i = i + 1

# %%
diverge_from_gradient = 2.5
diverge_from_y_intercept = 1.0

# TODO: from left-edge or something here
diverge_from = SplineScipy(
    scipy.interpolate.PPoly(
        c=[
            [diverge_from_gradient],
            [diverge_from_y_intercept - 10.0 * diverge_from_gradient],
        ],
        x=[-10.0, 10.0],
    )
)

# %% [markdown]
# ### Harmonisation time > convergence time
# In the following, we consider the same nine scenarios as
# above in which the `harmonisee` spline differs
# from the `diverge_from` spline due to varying shifts in the
# intercept ([0.0, -1.2, 1.2]) and slope ([1.0, 0.7, 1.4]).
# However, this time we consider in all upcoming scenarios
# harmonisation time (=1.0) > convergence time (=-1.0).

# %%
harmonisation_time = 1.0
convergence_time = -1.0

# %%
# Backwards along x harmonisation
i = 0
for y_intercept_shift in [0.0, -1.2, 1.2]:
    for gradient_factor in [1.0, 0.7, 1.4]:
        harmonisee = SplineScipy(
            scipy.interpolate.PPoly(
                c=[
                    [diverge_from_gradient * gradient_factor],
                    [
                        diverge_from_y_intercept
                        - 10.0 * diverge_from_gradient
                        + y_intercept_shift
                    ],
                ],
                x=[-10.0, 10.0],
            )
        )

        res = harmonise_splines_add_cubic(
            diverge_from=diverge_from,
            harmonisee=harmonisee,
            harmonisation_time=harmonisation_time,
            convergence_time=convergence_time,
        )

        fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

        plot_spline(
            diverge_from, np.linspace(-1.0, 3.0, 101), ax=axes[0], label="diverge_from"
        )
        plot_spline(
            harmonisee,
            np.linspace(harmonisation_time, 2 * convergence_time, 101),
            ax=axes[0],
            label="harmonisee",
        )
        plot_spline(
            res,
            np.linspace(harmonisation_time, 2 * convergence_time, 101),
            ax=axes[0],
            label="res",
        )

        plot_spline(
            diverge_from.derivative(),
            np.linspace(-1.0, 3.0, 101),
            ax=axes[1],
            label="diverge_from",
            gradient=True,
        )
        plot_spline(
            harmonisee.derivative(),
            np.linspace(harmonisation_time, 2 * convergence_time, 101),
            ax=axes[1],
            label="harmonisee",
            gradient=True,
        )
        plot_spline(
            res.derivative(),
            np.linspace(harmonisation_time, 2 * convergence_time, 101),
            ax=axes[1],
            label="cubic-spline",
            gradient=True,
        )

        for ax in axes:
            ax.axvline(
                harmonisation_time,
                label="harmonisation_time",
                color="gray",
                linestyle=":",
            )
            ax.axvline(
                convergence_time, label="convergence_time", color="gray", linestyle="--"
            )
        for ax in axes[1::2]:
            ax.legend(handlelength=1.1, loc="center right", fontsize="small")

        fig.suptitle(
            f"Scenario {i+1} (intercept shift: {y_intercept_shift},"
            + f" slope factor: {gradient_factor})"
        )
        plt.show()
        i = i + 1
