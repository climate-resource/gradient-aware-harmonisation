"""
Gradient-aware harmonisation of timeseries
"""

import importlib.metadata
from typing import Any, Optional, Union

from gradient_aware_harmonisation import utils
from gradient_aware_harmonisation.utils import (
    Timeseries,
    biased_corrected_harmonisee,
    compute_splines,
    harmonise_splines,
    interpolate_harmoniser,
)

__version__ = importlib.metadata.version("gradient_aware_harmonisation")


def harmonise(  # noqa: PLR0913
    target_timeseries: Timeseries,
    harmonisee_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_time: Optional[Union[int, float]],
    interpolation_target: str = "original",
    decay_method: str = "cosine",
    **kwargs: Any,
) -> Timeseries:
    """
    Harmonise two timeseries

    When we say harmonise, we mean make it
    such that the harmonisee matches with the target at some
    specified time point (called harmonisation time)

    Parameters
    ----------
    target_timeseries
        Target timeseries (i.e. what we harmonise to)

    harmonisee_timeseries
        Harmonisee timeseries (i.e. the timeseries we want to harmonise)

    harmonisation_time
        Time point at which harmonisee should be matched to the target

    convergence_time
        Time point at which the harmonised data
        should converge towards the prediced data.

    interpolation_target
        Target to which the harmonised timeseries should converge.

        If original, we converge back to harmonisee.
        If bias-corrected, we converge back to harmonissee
        having applied a basic constant offset bias correction
        (see the docs for further info TODO put a cross link to a notebook).

    decay_method
        Decay function used to decay weights
        when interpolating between the target and our harmonisation target.
        If 'polynomial' is used an additional argument 'pow' to specify the power
        is required (should be => 1.)

    **kwargs
        keyword arguments passed to `make_interp_spline` or 'polynomial_decay'

    Returns
    -------
    harmonised_timeseries :
        timeseries of harmonised data set

    Raises
    ------
    ValueError
        interpolation_target must be either 'original' or 'bias_corrected'
    """
    if interpolation_target not in ["original", "bias_corrected"]:
        raise ValueError(  # noqa: TRY003
            "interpolation_target must be 'original' or 'bias_corrected'. "
            f"Got {interpolation_target=}"
        )

    # compute splines
    splines = compute_splines(
        target=target_timeseries, harmonisee=harmonisee_timeseries, **kwargs
    )

    # compute harmonised spline
    harmonised_spline = harmonise_splines(
        target=splines["target"],
        harmonisee=splines["harmonisee"],
        harmonisation_time=harmonisation_time,
        **kwargs,
    )

    # get target of interpolation
    if interpolation_target == "original":
        interpol_target = splines["harmonisee"]
    if interpolation_target == "bias_corrected":
        interpol_target = biased_corrected_harmonisee(
            splines=splines, harmonisation_time=harmonisation_time, **kwargs
        )

    # compute interpolation timeseries
    interpolated_timeseries = interpolate_harmoniser(
        interpol_target,
        harmonised_spline,
        harmonisee_timeseries,
        convergence_time,
        harmonisation_time,
        decay_method,
        **kwargs,
    )

    return interpolated_timeseries


__all__ = [
    "harmonise",
    "utils",
]
