import numpy as np
import pandas as pd
import scipy as sp
import scipy.interpolate

spi = sp.interpolate

from typing import Optional, Union


class Timeseries:
    time_axis: np.array
    value: np.array


class Spline:
    scipy.interpolate.make_interp_spline


class SplinesCollection:
    target: Spline
    harmonisee: Spline


def convert_to_timeseries(time_axis: np.array, values: np.array) -> Timeseries:
    """
    Converts input into timeseries object

    Parameters
    ----------
    time_axis: np.array
        sequence of values related to the time axies (e.g., years)
    values: np.array
        sequence of values related to the measurements (e.g., C02)

    Returns
    -------
    timeseries: Timeseries
        timeseries object with time_axis and value attribute
    """
    timeseries: Timeseries = pd.DataFrame({"time_axis": time_axis, "value": values})
    return timeseries


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
    return spi.make_interp_spline(
        timeseries.time_axis.values, timeseries.value.values, **kwargs
    )


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

    harmonised_timeseries: Timeseries = pd.DataFrame(
        dict(time_axis=timeseries_harmonisee.time_axis.values, value=harmonised_values)
    )

    return harmonised_timeseries


def find_index_convergence_time(
    timeseries: Timeseries, harmonisation_time: Union[int, float]
) -> int:
    """
    Finds the index of the harmonised time series corresponding to the given timeseries.

    Parameters
    ----------
    timeseries : Timeseries
        timeseries of format dict(time_axis = np.array, values = np.array)
    harmonisation_time : Union[int, float]
        point in time_axis at which harmonise should be matched to target

    Returns
    -------
    found_index : int
        index of the timeseries corresponding to the harmonisation time.
    """
    time_values = timeseries.time_axis.values
    # default value
    check = False
    for i in range(len(time_values) - 1):
        # pass check (index found)
        check = False
        if (
            time_values[i] <= harmonisation_time
            and time_values[i + 1] >= harmonisation_time
        ):
            check = True
            found_index = i
            break
    if check is not True:
        found_index = None
        raise ValueError(
            f"The provided harmonisation_time={harmonisation_time} is not covered by both provided timeseries."
        )
    return found_index


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

    total_time_range = len(timeseries_harmonisee.time_axis.values)
    idx0 = find_index_convergence_time(timeseries_harmonisee, harmonisation_time)
    if convergence_time is None:
        time_axis = timeseries_harmonisee.time_axis.values[idx0:]
        decay_range = len(time_axis)
        fill_with_zeros = []
    else:
        idx1 = find_index_convergence_time(timeseries_harmonisee, convergence_time)
        time_axis = timeseries_harmonisee.time_axis.values[idx0:idx1]
        decay_range = len(time_axis)
        # get length of decay sequence; add zeros to fillup remaining time_axis
        diff_len = total_time_range - idx1
        fill_with_zeros = [0.0] * diff_len

    # decay function
    if decay_method == "cosine":
        raise NotImplementedError
        # decay_function = tf.keras.optimizers.schedules.CosineDecay(1.0, decay_range)

    # compute weight
    weight_seq = [decay_function(weight) for weight in range(decay_range)]
    weight_sequence = np.stack(weight_seq + fill_with_zeros)

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
    idx0 = find_index_convergence_time(timeseries_harmonisee, harmonisation_time)

    updated_time_axis, values_interpolated = [], []

    for i, w in enumerate(decay_weights):
        #  time_index = len(decay_weights)-i
        updated_time_axis.append(timeseries_harmonisee.time_axis.values[idx0 + i])
        values_interpolated.append(
            w * harmonised(updated_time_axis[-1])
            + (1 - w) * harmonisee(updated_time_axis[-1])
        )

    timeseries_interpolated: Timeseries = pd.DataFrame(
        dict(
            time_axis=updated_time_axis,
            value=values_interpolated,
        )
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
