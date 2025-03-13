from typing import Optional, Union

import numpy as np
import pandas as pd
from attrs import define, field


@define
class Timeseries:
    """
    Timeseries class
    """

    time_axis: np.array
    values: np.array = field()

    @values.validator
    def values_validator(self, attribute, value):
        if value.size != self.time_axis.size:
            msg = (
                f"{attribute.name} must have the same size as time_axis. "
                f"Received {value.size=} {self.time_axis.size=}"
            )
            raise ValueError(msg)


class Spline:
    scipy.interpolate.make_interp_spline


class SplinesCollection:
    target: Spline
    harmonisee: Spline


def timeseries_to_spline(timeseries: Timeseries, **kwargs) -> Spline:
    """
    Estimates splines from timeseries arrays.

    Parameters
    ----------
    timeseries : Timeseries
        timeseries of format dict(time_axis = np.array, values = np.array)

    **kwargs :
        additional arguments to ``scipy.interpolate.make_interp_spline``

    Returns
    -------
    spline : Spline
        compute spline from timeseries data
    """
    return spi.make_interp_spline(timeseries.time_axis, timeseries.values, **kwargs)


def derivative(spline: Spline) -> Spline:
    """
    Computes the first derivative of the passed spline function.

    Parameters
    ----------
    spline : Spline
        spline function from data array

    Returns
    -------
    spline : Spline
        1st derivative of spline

    """
    return spline.derivative()


def integrate(spline: Spline) -> Spline:
    """
    Compute the antiderivative of the passed spline function.

    Parameters
    ----------
    spline : Spline
        spline function from data array

    Returns
    -------
    spline : Spline
        antiderivative of spline
    """
    return spline.antiderivative()


def harmonise_timeseries(
    target: Spline,
    harmonisee: Spline,
    timeseries_harmonisee: Timeseries,
    harmonisation_time: Union[int, float],
) -> Timeseries:
    """
    Computes a timeseries based on the adjustment of the harmonisee to the target.

    Parameters
    ----------
    target: Spline
        target spline from timeseries array
    harmonisee: Spline
        harmonisee spline from timeseries array (should be adjusted to target spline)
    timeseries_harmonisee: Timeseries
        harmonisee timeseries of format dict(time_axis = np.array, values = np.array)
    harmonisation_time: Union[int, float]
        point in time_axis at which harmonisee should be matched to target

    Returns
    -------
    harmonised_timeseries: Timeseries
        harmonised timeseries
    """
    diff = target(harmonisation_time) - harmonisee(harmonisation_time)
    harmonised_values = harmonisee(timeseries_harmonisee.time_axis) + diff

    harmonised_timeseries = Timeseries(
        time_axis=timeseries_harmonisee.time_axis,
        values=harmonised_values,
    )

    return harmonised_timeseries


def decay_weights(
    timeseries_harmonisee: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_time: Optional[Union[int, float]],
    decay_method: str,
) -> np.array:
    """
    Compute a sequence of decaying weights according to specified decay method.

    Parameters
    ----------
    timeseries_harmonisee : Timeseries
        timeseries of harmonised spline
    harmonisation_time: Union[int, float]
        point in time_axis at which harmonise should be matched to target
    convergence_time : Union[int, float]
        time point at which harmonisee should match target function
    decay_method : str
        decay method to use

    Returns
    -------
    weight_sequence : np.array
        sequence of weights for interpolation

    Raises
    ------
    ValueError
        Currently supported values for `decay_method` are: "cosine"
    """
    if decay_method not in ["cosine"]:
        raise ValueError(
            f"Currently supported values for `decay_method` are 'cosine'. Got {decay_method}."
        )

    if not np.isin(harmonisation_time, timeseries_harmonisee.time_axis).all():
        msg = (
            f"{harmonisation_time=} is not a value in "
            f"{timeseries_harmonisee.time_axis=}"
        )
        raise NotImplementedError(msg)

    if convergence_time is None:
        time_interp = timeseries_harmonisee.time_axis[
            np.where(timeseries_harmonisee.time_axis >= harmonisation_time)
        ]
        # decay_range = len(time_axis)
        fill_with_zeros = []

    else:
        time_interp = timeseries_harmonisee.time_axis[
            np.where(
                np.logical_and(
                    timeseries_harmonisee.time_axis >= harmonisation_time,
                    timeseries_harmonisee.time_axis <= convergence_time,
                )
            )
        ]

        time_match_harmonisee = timeseries_harmonisee.time_axis[
            np.where(timeseries_harmonisee.time_axis > convergence_time)
        ]
        fill_with_zeros = np.zeros_like(time_match_harmonisee)

    # decay function
    if decay_method == "cosine":
        # TODO: fix this, obviously wrong
        weight_seq = np.ones_like(time_interp)

    # compute weight
    weight_sequence = np.concatenate((weight_seq, fill_with_zeros))

    return weight_sequence


def interpolate_timeseries(
    harmonisee: Spline,
    harmonised: Spline,
    harmonisation_time: Union[int, float],
    timeseries_harmonisee: Timeseries,
    decay_weights: np.array,
) -> Timeseries:
    """
    Computes timeseries that interpolates between harmonised spline at harmonisation time and target spline at either
    the last date of the harmonisee or the specified convergence time.

    Parameters
    ----------
    harmonisee : Spline
        harmonisee spline
    harmonised : Spline
        harmonised (adjusted) spline
    harmonisation_time: Union[int, float]
        time point at which harmonisee and target should match
    timeseries_harmonisee : Timeseries
        timeseries of the harmonisee
    decay_weights : np.array
        sequence of weights decaying from 1 to 0

    Returns
    -------
    timeseries_interpolated : Timeseries
        timeseries that interpolate between harmonised spline and harmonisee
    """
    # timeseries harmonised
    # timeseries_harmonised = harmonised(timeseries_harmonisee.time_axis.values)
    # reduce timeseries from harmonisation time point

    if not np.isin(harmonisation_time, timeseries_harmonisee.time_axis).all():
        msg = (
            f"{harmonisation_time=} is not a value in "
            f"{timeseries_harmonisee.time_axis=}"
        )
        raise NotImplementedError(msg)

    updated_time_axis = timeseries_harmonisee.time_axis[
        np.where(timeseries_harmonisee.time_axis >= harmonisation_time)
    ]
    harmonised_values = harmonised(updated_time_axis)
    harmonisee_values = harmonisee(updated_time_axis)
    values_interpolated = (
        decay_weights * harmonised_values + (1 - decay_weights) * harmonisee_values
    )

    timeseries_interpolated = Timeseries(
        time_axis=updated_time_axis,
        values=values_interpolated,
    )

    return timeseries_interpolated


# %% Wrapper
def compute_splines(
    target: Timeseries, harmonisee: Timeseries, **kwargs
) -> SplinesCollection:
    """
    Converts input arrays into timeseries objects and computes splines

    Parameters
    ----------
    target : Timeseries
        Timeseries of target data
    harmonisee : Timeseries
        timeseries of matching data (have to be adjusted to match the target)
    **kwargs
        keyword arguments passed to make_interp_spline

    Returns
    -------
    splines : SplinesCollection
        splines of target and harmonisee
    """
    # compute splines
    target_spline = timeseries_to_spline(target, **kwargs)
    harmonisee_spline = timeseries_to_spline(harmonisee, **kwargs)

    splines: SplinesCollection = pd.DataFrame(
        dict(target=[target_spline], harmonisee=[harmonisee_spline])
    )
    return splines


def interpolate_harmoniser(
    interpolation_target: Spline,
    harmonised_spline: Spline,
    harmonisee_timeseries: Timeseries,
    convergence_time: Optional[Union[int, float]],
    harmonisation_time: Union[int, float],
    decay_method: str = "cosine",
) -> Timeseries:
    """
    Computes an interpolated timeseries which interpolates from the harmonised_spline to the interpolation target

    Parameters
    ----------
    interpolation_target : Spline
        interpolation target, i.e., with which predicitons should the interpolation spline match after the convergence
         time? Usually this will be either the original harmonisee or the biased-corrected harmonisee
    harmonised_spline : Spline
        harmonised spline that matches with target wrt zero-and first-order derivative
    harmonisee_timeseries : Timeseries
        harmonisee timeseries
    convergence_time : Optional[Union[int, float]]
        time point where interpolation_target and harmonised spline should match
    harmonisation_time : Union[int, float]
        time point where harmonised spline should match the original target
    decay_method : str, default="cosine"
        decay method used for computing weights that interpolate the spline, currently supported methods are 'cosine'.

    Returns
    -------
    interpolated_timeseries : Timeseries
        interpolated values
    """
    # get interpolation weights
    weights = decay_weights(
        harmonisee_timeseries,
        convergence_time=convergence_time,
        harmonisation_time=harmonisation_time,
        decay_method=decay_method,
    )

    # compute interpolation spline
    interpolated_timeseries = interpolate_timeseries(
        interpolation_target,
        harmonised_spline,
        harmonisation_time,
        harmonisee_timeseries,
        weights,
    )

    return interpolated_timeseries


def harmonise_splines(
    splines: SplinesCollection,
    harmonisee_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    **kwargs,
) -> Spline:
    """
    Harmonises two splines by matching a harmonisee to a target spline

    Parameters
    ----------
    splines : SplinesCollection
        splines of target and harmonisee as computed by :func:`compute_splines`
    harmonisee_timeseries : Timeseries
        timeseries of matching data
    harmonisation_time : Union[int, float]
        time point at which harmonisee should be matched to the target
    **kwargs
        keyword arguments passed to make_interp_spline

    Returns
    -------
    harmonised_spline : Spline
        harmonised spline (harmonised spline and target have same zero-and first-order derivative at harmonisation time)
    """
    # compute derivatives
    target_dspline = derivative(splines.target[0])
    harmonisee_dspline = derivative(splines.harmonisee[0])

    # match first-order derivatives
    harmonised_d1_timeseries = harmonise_timeseries(
        target_dspline, harmonisee_dspline, harmonisee_timeseries, harmonisation_time
    )
    # compute spline
    harmonised_D1_spline = timeseries_to_spline(harmonised_d1_timeseries, **kwargs)
    # integrate to match zero-order derivative
    harmonised_d1_spline = integrate(harmonised_D1_spline)

    # match zero-order derivatives
    harmonised_d0d1_timeseries = harmonise_timeseries(
        splines.target[0],
        harmonised_d1_spline,
        harmonisee_timeseries,
        harmonisation_time,
    )
    # compute spline
    harmonised_d0d1_spline = timeseries_to_spline(harmonised_d0d1_timeseries, **kwargs)

    return harmonised_d0d1_spline


def biased_corrected_harmonisee(
    splines: SplinesCollection,
    harmonisee_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    **kwargs,
) -> Spline:
    """
    Computes the biased corrected spline, i.e. the harmonisee matches the target spline wrt the zero-order
    derivative.

    Parameters
    ----------
    splines : SplinesCollection
        splines of target and harmonisee as computed by :func:`compute_splines`
    harmonisee_timeseries : Timeseries
        timeseries of matching data
    harmonisation_time : Union[int, float]
        time point at which harmonisee should be matched to the target
    **kwargs
        keyword arguments passed to make_interp_spline

    Returns
    -------
    biased_corrected_spline : Spline
        biased corrected spline
    """
    biased_corrected_timeseries = harmonise_timeseries(
        splines.target[0],
        splines.harmonisee[0],
        harmonisee_timeseries,
        harmonisation_time,
    )
    biased_corrected_spline = timeseries_to_spline(
        biased_corrected_timeseries, **kwargs
    )

    return biased_corrected_spline
