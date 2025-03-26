"""
Gradient-aware harmonisation of timeseries
"""

import importlib.metadata
from typing import Optional, Union

from gradient_aware_harmonisation.timeseries import Timeseries
from gradient_aware_harmonisation.utils import (
    ConvergenceMethod,
    harmonise_splines,
)

__version__ = importlib.metadata.version("gradient_aware_harmonisation")


def harmonise(  # noqa: PLR0913
    harmonisee_timeseries: Timeseries,
    target_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_timeseries: Timeseries | None = None,
    convergence_time: Optional[Union[int, float]] | None = None,
    convergence_method: ConvergenceMethod = ConvergenceMethod.COSINE,
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

    harmonised_spline = harmonise_splines(
        harmonisee=harmonisee_timeseries.to_spline(),
        target=target_timeseries.to_spline(),
        harmonisation_time=harmonisation_time,
        convergence_spline=convergence_timeseries.to_spline(),
        convergence_time=convergence_time,
        convergence_method=convergence_method,
    )

    harmonised_timeseries = harmonised_spline.to_timeseries(
        time_axis=harmonisee_timeseries.time_axis
    )

    return harmonised_timeseries


__all__ = ["harmonise"]
