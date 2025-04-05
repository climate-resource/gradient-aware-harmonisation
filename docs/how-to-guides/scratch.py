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

# %%
from __future__ import annotations

import numpy as np
import pytest
import scipy.interpolate
import matplotlib.pyplot as plt
from gradient_aware_harmonisation.add_cubic import (
    harmonise_splines_add_cubic,
    taylor_shift,
)
from gradient_aware_harmonisation.spline import Spline, SplineScipy

# %%
diverge_from_gradient = 2.5
diverge_from_y_intercept = 1.0

diverge_from = SplineScipy(
    scipy.interpolate.PPoly(
        c=[[diverge_from_gradient], [diverge_from_y_intercept]],
        x=[0, 1e8],
    )
)


# %%
def plot_spline(spline, x, ax, label):
    ax.plot(
        x,
        spline(x),
        label=label,
    )


# %%
harmonisation_time = 0.0
convergence_time = 3.2

# %%
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
        
        plot_spline(diverge_from, np.linspace(-1.0, 3.0, 101), ax=axes[0], label="diverge_from")
        plot_spline(harmonisee, np.linspace(harmonisation_time, 2 * convergence_time, 101), ax=axes[0], label="harmonisee")
        plot_spline(res, np.linspace(harmonisation_time, 2 * convergence_time, 101), ax=axes[0], label="res")

        plot_spline(diverge_from.derivative(), np.linspace(-1.0, 3.0, 101), ax=axes[1], label="diverge_from_gradient")
        plot_spline(harmonisee.derivative(), np.linspace(harmonisation_time, 2 * convergence_time, 101), ax=axes[1], label="harmonisee_gradient")
        plot_spline(res.derivative(), np.linspace(harmonisation_time, 2 * convergence_time, 101), ax=axes[1], label="res_gradient")

        for ax in axes:
            ax.axvline(harmonisation_time, label="harmonisation_time", color="gray", linestyle=":")
            ax.axvline(convergence_time, label="convergence_time", color="gray", linestyle="--")
            ax.legend()

# %%
diverge_from_gradient = 2.5
diverge_from_y_intercept = 1.0

# TODO: from left-edge or something here
diverge_from = SplineScipy(
    scipy.interpolate.PPoly(
        c=[[diverge_from_gradient], [diverge_from_y_intercept - 10.0 * diverge_from_gradient]],
        x=[-10.0, 10.0],
    )
)

# %%
harmonisation_time = 1.0
convergence_time = -1.0

# %%
# Backwards along x harmonisation
for y_intercept_shift in [0.0, -1.2, 1.2]:
    for gradient_factor in [1.0, 0.7, 1.4]:
        harmonisee = SplineScipy(
            scipy.interpolate.PPoly(
                c=[
                    [diverge_from_gradient * gradient_factor],
                    [diverge_from_y_intercept - 10.0 * diverge_from_gradient + y_intercept_shift],
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
        
        plot_spline(diverge_from, np.linspace(-1.0, 3.0, 101), ax=axes[0], label="diverge_from")
        plot_spline(harmonisee, np.linspace(harmonisation_time, 2 * convergence_time, 101), ax=axes[0], label="harmonisee")
        plot_spline(res, np.linspace(harmonisation_time, 2 * convergence_time, 101), ax=axes[0], label="res")

        plot_spline(diverge_from.derivative(), np.linspace(-1.0, 3.0, 101), ax=axes[1], label="diverge_from_gradient")
        plot_spline(harmonisee.derivative(), np.linspace(harmonisation_time, 2 * convergence_time, 101), ax=axes[1], label="harmonisee_gradient")
        plot_spline(res.derivative(), np.linspace(harmonisation_time, 2 * convergence_time, 101), ax=axes[1], label="res_gradient")

        for ax in axes:
            ax.axvline(harmonisation_time, label="harmonisation_time", color="gray", linestyle=":")
            ax.axvline(convergence_time, label="convergence_time", color="gray", linestyle="--")
            ax.legend()
