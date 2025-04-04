"""
Gradient-aware harmonisation of timeseries
"""

import importlib.metadata
from typing import Optional, Union

from gradient_aware_harmonisation.convergence import get_cosine_decay_harmonised_spline
from gradient_aware_harmonisation.timeseries import Timeseries
from gradient_aware_harmonisation.utils import (
    GetHarmonisedSplineLike,
    harmonise_splines,
)

__version__ = importlib.metadata.version("gradient_aware_harmonisation")


def harmonise(  # noqa: PLR0913
    harmonisee_timeseries: Timeseries,
    target_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_timeseries: Optional[Timeseries] = None,
    convergence_time: Optional[Union[int, float]] = None,
    get_harmonised_spline: GetHarmonisedSplineLike = (
        get_cosine_decay_harmonised_spline
    ),
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

    get_harmonised_spline
        Function used to get the harmonised spline from a gradient-
        preserving spline and the timeseries to converge to

    Returns
    -------
    harmonised_timeseries :
        Harmonised timeseries
    """
    # use maximum time if no convergence time is provided
    if convergence_time is None:
        convergence_time = harmonisee_timeseries.time_axis.max()

    # use harmonisee as convergence target if nothing else is provided
    if convergence_timeseries is None:
        convergence_timeseries = harmonisee_timeseries

    # convert timeseries to splines
    target_spline = target_timeseries.to_spline()
    harmonisee_spline = harmonisee_timeseries.to_spline()
    convergence_spline = convergence_timeseries.to_spline()

    # get harmonised spline
    harmonised_spline = harmonise_splines(
        harmonisee=harmonisee_spline,
        target=target_spline,
        harmonisation_time=harmonisation_time,
        converge_to=convergence_spline,
        convergence_time=convergence_time,
        get_harmonised_spline=get_harmonised_spline,
    )

    # convert harmonised spline to timeseries
    res_time_axis = harmonisee_timeseries.time_axis[
        harmonisee_timeseries.time_axis >= harmonisation_time
    ]
    res = Timeseries(
        time_axis=res_time_axis,
        values=harmonised_spline(res_time_axis),
    )

    return res


__all__ = [
    "harmonise",
    "harmonise_splines",
]
