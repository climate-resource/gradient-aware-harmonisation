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
# - take some splines
# - harmonise them with default convergence
# - harmonise them with different convergence
# - plots

# %%
import matplotlib.pyplot as plt
import numpy as np

from gradient_aware_harmonisation.timeseries import Timeseries
from gradient_aware_harmonisation.utils import harmonise_splines

# %%
target_ts = Timeseries(
    time_axis=np.array([2000, 2001, 2002, 2003]),
    values=np.array([371.77, 373.72, 376.33, 378.43]),
)
target = target_ts.to_spline(k=1)

harmonisee_ts = Timeseries(
    time_axis=np.array([2003, 2004, 2005, 2006]),
    values=np.array([376.28, 378.83, 381.20, 382.55]),
)
harmonisee = harmonisee_ts.to_spline(k=2)

# %%
fig, ax = plt.subplots()

x_axis = np.linspace(target_ts.time_axis.min(), target_ts.time_axis.max(), 500)
ax.plot(x_axis, target(x_axis), label="target")
ax.plot(
    harmonisee_ts.time_axis, harmonisee(harmonisee_ts.time_axis), label="harmonisee"
)
ax.legend()

# %%
harmonised_spline = harmonise_splines(
    harmonisee=harmonisee,
    target=target,
    harmonisation_time=2003,
    convergence_spline=harmonisee,
    convergence_time=2006,
)

# %%
fig, ax = plt.subplots()

x_axis = np.linspace(target_ts.time_axis.min(), target_ts.time_axis.max(), 500)
ax.plot(x_axis, target(x_axis), label="target")

x_axis = np.linspace(harmonisee_ts.time_axis.min(), harmonisee_ts.time_axis.max(), 500)
ax.plot(x_axis, harmonisee(x_axis), label="harmonisee")
ax.plot(x_axis, [harmonised_spline(v) for v in x_axis], label="harmonised")

ax.legend()

# %%
fig, ax = plt.subplots()

ax.plot(target_ts.time_axis, target(target_ts.time_axis), label="target")
ax.plot(
    harmonisee_ts.time_axis, harmonisee(harmonisee_ts.time_axis), label="harmonisee"
)
ax.plot(
    harmonisee_ts.time_axis,
    [harmonised_spline(v) for v in harmonisee_ts.time_axis],
    label="harmonised",
)

ax.legend()

# %%
from typing import Union

from attrs import define

from gradient_aware_harmonisation.spline import Spline


@define
class SplinePolynomialConvergence:
    initial_time: Union[float, int]
    """
    At and before this time, we return values from `initial`
    """

    final_time: Union[float, int]
    """
    At and after this time, we return values from `final`
    """

    initial: Spline
    """
    The spline whose values we use at and before `initial_time`
    """

    final: Spline
    """
    The spline whose values we use at and after `final_time`
    """

    power: int
    """Degree of the polynomial"""

    def __call__(self, x: int | float) -> int | float:
        """
        Evaluate the spline at a given x-value

        Parameters
        ----------
        x
            x-value

        Returns
        -------
        :
            Value of the spline at `x`
        """
        if x <= self.initial_time:
            return self.initial(x)

        if x >= self.final_time:
            return self.final(x)

        gamma = (
            1 - (self.final_time - x) / (self.final_time - self.initial_time)
        ) ** self.power
        res = gamma * self.initial(x) + (1 - gamma) * self.final(x)

        return res

    def derivative(self) -> None:
        raise NotImplementedError

    def antiderivative(self) -> None:
        raise NotImplementedError


def get_polynomial_decay_harmonised_spline(
    harmonisation_time: Union[int, float],
    convergence_time: Union[int, float],
    harmonised_spline_no_convergence: Spline,
    convergence_spline: Spline,
    power: int,
) -> SplinePolynomialConvergence:
    return SplinePolynomialConvergence(
        initial_time=harmonisation_time,
        final_time=convergence_time,
        initial=harmonised_spline_no_convergence,
        final=convergence_spline,
        power=power,
    )


# %%
from functools import partial

power_convergence_d = {}
for power in [1, 2, 3, 4]:
    harmonised_spline = harmonise_splines(
        harmonisee=harmonisee,
        target=target,
        harmonisation_time=2003,
        convergence_spline=harmonisee,
        convergence_time=2006,
        get_harmonised_spline=partial(
            get_polynomial_decay_harmonised_spline, power=power
        ),
    )

    power_convergence_d[power] = harmonised_spline

# %%
fig, ax = plt.subplots()

ax.plot(target_ts.time_axis, target(target_ts.time_axis), label="target")
ax.plot(
    harmonisee_ts.time_axis, harmonisee(harmonisee_ts.time_axis), label="harmonisee"
)
ax.plot(
    harmonisee_ts.time_axis,
    [harmonised_spline(v) for v in harmonisee_ts.time_axis],
    label="harmonised",
)
for power, harmonised_spline in power_convergence_d.items():
    ax.plot(
        harmonisee_ts.time_axis,
        [harmonised_spline(v) for v in harmonisee_ts.time_axis],
        label=f"harmonised-polynomial-degree-{power}",
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
