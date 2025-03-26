"""
Gradient-aware harmonisation of timeseries
"""

import importlib.metadata
from typing import Optional, Union

import numpy as np

from gradient_aware_harmonisation.convergence import SplineCosineConvergence
from gradient_aware_harmonisation.timeseries import Timeseries
from gradient_aware_harmonisation.utils import (
    GetHarmonisedSplineLike,
    # harmonise_splines,
    add_constant_to_spline,
)

__version__ = importlib.metadata.version("gradient_aware_harmonisation")


def harmonise(  # noqa: PLR0913
    harmonisee_timeseries: Timeseries,
    target_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_timeseries: Timeseries | None = None,
    convergence_time: Optional[Union[int, float]] | None = None,
    get_harmonised_spline: GetHarmonisedSplineLike = SplineCosineConvergence,
    # get_harmonised_spline: GetHarmonisedSplineLike = get_cosine_decay_harmonised_spline,
    # convergence_function: Callable[[Spline, Spline], Spline] | None = None,
) -> Timeseries:
    """
    Harmonise two timeseries

    When we say harmonise, we mean make it
    such that the harmonisee matches with the target at some
    specified time point (called harmonisation time)
    before returning to some other timeseries
    (the convergence timeseries)
    at the convergence time.

    Parameters
    ----------
    harmonisee_timeseries
        Harmonisee timeseries (i.e. the timeseries we want to harmonise)

    target_timeseries
        Target timeseries (i.e. what we harmonise to)

    harmonisation_time
        Time point at which harmonisee should be matched to the target

    convergence_timeseries
        The timeseries to which the result should converge.

        If not supplied, we use `harmonisee_timeseries`
        i.e. we converge back to the timeseries we are harmonising.

    convergence_time
        Time point at which the harmonised data
        should converge to the convergence timeseries.

        If not supplied, we converge to convergence timeseries
        at the last time point in harmonisee_timeseries.

    convergence_method
        The method to use to converge back to the convergence timeseries.

    Returns
    -------
    harmonised_timeseries :
        Harmonised timeseries
    """
    if convergence_time is None:
        convergence_time = harmonisee_timeseries.time_axis.max()

    # from timeseries to splines
    target_spline = target_timeseries.to_spline()
    harmonisee_spline = harmonisee_timeseries.to_spline()
    convergence_spline = convergence_timeseries.to_spline()

    # compute derivatives of splines
    target_dspline = target_spline.derivative()
    harmonisee_dspline = harmonisee_spline.derivative()

    # match first-order derivatives
    diff_dspline = np.subtract(
        target_dspline(harmonisation_time), harmonisee_dspline(harmonisation_time)
    )

    harmonised_first_derivative = add_constant_to_spline(
        in_spline=harmonisee_dspline, constant=diff_dspline
    )

    # integrate to match zero-order derivative
    harmonised_spline_first_derivative_only = (
        harmonised_first_derivative.antiderivative()
    )

    # match zero-order derivatives
    diff_spline = np.subtract(
        target_spline(harmonisation_time),
        harmonised_spline_first_derivative_only(harmonisation_time),
    )

    harmonised_spline_no_convergence = add_constant_to_spline(
        in_spline=harmonised_spline_first_derivative_only, constant=diff_spline
    )

    harmonised_spline = get_harmonised_spline(
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
        harmonised_spline_no_convergence=harmonised_spline_no_convergence,
        convergence_spline=convergence_spline,
    )

    return harmonised_spline(harmonisee_timeseries.time_axis)


__all__ = [
    "harmonise",
    # "harmonise_splines",
]
