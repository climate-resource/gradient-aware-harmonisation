from typing import Optional, Union

from gradient_aware_harmonisation.utils import (
    Timeseries,
    biased_corrected_harmonisee,
    compute_splines,
    harmonise_splines,
    interpolate_harmoniser,
)


def harmoniser(
    target_timeseries: Timeseries,
    harmonisee_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_time: Optional[Union[int, float]],
    interpolation_target: str = "original",
    decay_method: str = "cosine",
    **kwargs,
):
    """
    Computes the harmonisation of two timeseries such that the harmonisee matches with the target at some
    specified time point (called harmonisation time)

    Parameters
    ----------
    target_timeseries : Timeseries
        timeseries of target data (to which the harmonisee should be matched)
    harmonisee_timeseries : Timeseries
        timeseries of matching data (that should be matched to the target data at harmonisation time)
    harmonisation_time : Union[int, float]
        time point at which harmonisee should be matched to the target
    convergence_time : Optional[Union[int, float]]
        time point at which the harmonised data should converge towards the prediced data
    interpolation_target : str, ["original", "bias_corrected"]
        to which target the interpolated spline should converge: the original predicted data (harmonisee) or the
        bias-corrected harmonisee (match at harmonisation time wrt zero-order derivative)
    decay_method : str, ["cosine"]
        decay function used to decay weights in the weighted sum of the interpolation spline
    **kwargs
        keyword arguments passed to make_interp_spline
    """
    if interpolation_target not in ["original", "bias_corrected"]:
        raise ValueError(
            f"interpolation_target must be 'original' or 'bias_corrected'. Got {interpolation_target}"
        )

    # compute splines
    splines = compute_splines(
        target=target_timeseries, harmonisee=harmonisee_timeseries, **kwargs
    )

    # compute harmonised spline
    harmonised_spline = harmonise_splines(
        splines, harmonisee_timeseries, harmonisation_time, **kwargs
    )

    # get target of interpolation
    if interpolation_target == "original":
        interpol_target = splines.harmonisee[0]
    if interpolation_target == "bias_corrected":
        interpol_target = biased_corrected_harmonisee(
            splines, harmonisee_timeseries, harmonisation_time, **kwargs
        )

    # compute interpolation timeseries
    interpolated_timeseries = interpolate_harmoniser(
        interpol_target,
        harmonised_spline,
        harmonisee_timeseries,
        convergence_time,
        harmonisation_time,
        decay_method,
    )

    return interpolated_timeseries
