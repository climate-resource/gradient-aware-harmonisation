"""
Utility functions
"""

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

import numpy as np
import numpy.typing as npt
from attrs import define, field

from gradient_aware_harmonisation.exceptions import MissingOptionalDependencyError
from gradient_aware_harmonisation.spline import (
    Spline,
    SplineScipy,
    add_constant_to_spline,
)

if TYPE_CHECKING:
    pass


@define
class Timeseries:
    """
    Timeseries class
    """

    time_axis: npt.NDArray[Any]
    values: npt.NDArray[Any] = field()

    @values.validator
    def values_validator(self, attribute: Any, value: Any) -> None:
        """
        Validate the values

        Parameters
        ----------
        attribute
            Attribute to validate

        value
            Value to validate
        """
        if value.size != self.time_axis.size:
            msg = (
                f"{attribute.name} must have the same size as time_axis. "
                f"Received {value.size=} {self.time_axis.size=}"
            )
            raise ValueError(msg)


def timeseries_to_spline(timeseries: Timeseries, **kwargs: Any) -> SplineScipy:
    """
    Estimates splines from timeseries arrays.

    Parameters
    ----------
    timeseries
        timeseries of format dict(time_axis = np.array, values = np.array)

    **kwargs
        additional arguments to ``scipy.interpolate.make_interp_spline``

    Returns
    -------
    spline :
        compute spline from timeseries data

    Raises
    ------
    ValueError
        Spline degree (`k`) is smaller or equal to length of time_axis
        in timeseries obj. Must be at least equal to spline degree.
        (Default spline degree is `k = 3`)
    """
    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "timeseries_to_spline", requirement="scipy"
        ) from exc

    # extract from kwargs arguments of make_interp_spline
    args_make_interp_spline = inspect.getfullargspec(
        scipy.interpolate.make_interp_spline
    ).args
    kwargs_spline: dict[str, Any] = {
        f"{key}": kwargs[key] for key in kwargs if key in args_make_interp_spline
    }

    if "k" in kwargs_spline:
        if len(timeseries.time_axis) <= kwargs_spline["k"]:
            raise ValueError(
                f"Spline degree {kwargs['k']} is smaller or equal to length "
                + f"of provided time_axis (={len(timeseries.time_axis)}) "
                + "in timeseries object.'"
                " \n But must be greater than spline degree.",
            )
    else:
        default_args = inspect.getfullargspec(
            scipy.interpolate.make_interp_spline
        ).defaults

        if default_args is not None:
            default_k = default_args.index(0)
            if len(timeseries.time_axis) <= default_k:
                raise ValueError(
                    f"Default spline degree k={default_k} is smaller or equal "
                    + "to length of provided time_axis "
                    + f"(={len(timeseries.time_axis)}) in timeseries object."
                    + "\n But must be greater than spline degree.",
                )

    spline = SplineScipy(
        scipy.interpolate.make_interp_spline(
            timeseries.time_axis, timeseries.values, **kwargs_spline
        )
    )

    return spline


def harmonise_constant_offset(
    target: Spline,
    harmonisee: Spline,
    harmonisation_time: Union[int, float],
) -> Spline:
    """
    Harmonise timeseries using a constant offset

    In other words, the timeseries are harmonised
    by simply adding a constant to the harmonisee
    such that its value matches the value of the target at the harmonisation time.

    Parameters
    ----------
    target
        Target for harmonisation

    harmonisee
        Function/spline to harmonisation

    harmonisation_time
        Time at which `target` and `harmonisee` should match exactly

    Returns
    -------
    :
        Harmonised spline
    """
    diff = target(harmonisation_time) - harmonisee(harmonisation_time)
    harmonised = add_constant_to_spline(in_spline=harmonisee, constant=diff)

    return harmonised


def cosine_decay(decay_steps: int, initial_weight: float = 1.0) -> npt.NDArray[Any]:
    """
    Compute cosine decay function

    Parameters
    ----------
    decay_steps
        number of steps to decay over

    initial_weight
        starting weight with default = 1.

    Returns
    -------
    weight_seq :
        weight sequence

    Reference
    ---------
    + `cosine decay as implemented in tensorflow.keras <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>`_
    """
    # initialize weight sequence
    weight_seq: list[float] = []
    # loop over number of steps
    for step in range(decay_steps):
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / (decay_steps - 1)))
        weight_seq.append(initial_weight * cosine_decay)

    return np.concatenate((weight_seq,))


def polynomial_decay(
    decay_steps: int, pow: Union[float, int], initial_weight: float = 1.0
) -> npt.NDArray[Any]:
    """
    Compute polynomial decay function

    Parameters
    ----------
    decay_steps
        number of steps to decay over

    pow
        power of polynomial
        expected to be greater or equal to 1.

    initial_weight
        starting weight, default is 1.

    Returns
    -------
    weight_seq :
        weight sequence

    Raises
    ------
    ValueError
        Power of polynomial is expected to be greater or equal to 1.
    """
    if not pow >= 1.0:
        msg = (
            "Power of polynomial decay is expected to be greater than or equal to 1. ",
            f"Got {pow=}.",
        )
        raise ValueError(msg)

    # initialize weight sequence
    weight_seq: list[float] = []
    # loop over steps
    for step in range(decay_steps):
        weight = initial_weight * (1 - step / (decay_steps - 1)) ** pow
        weight_seq.append(weight)

    return np.concatenate((weight_seq,))


def decay_weights(
    timeseries_harmonisee: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_time: Optional[Union[int, float]],
    decay_method: str,
    **kwargs: Any,
) -> npt.NDArray[Any]:
    """
    Compute a sequence of decaying weights according to specified decay method.

    Parameters
    ----------
    timeseries_harmonisee
        timeseries of harmonised spline

    harmonisation_time
        point in time_axis at which harmonise should be matched to target

    convergence_time
        time point at which harmonisee should match target function

    decay_method
        decay method to use
        If decay_method="polynomial" power of the polynmials (arg: 'pow') is required;
        'pow' is expected to be greater or equal to 1.

    Returns
    -------
    weight_sequence :
        sequence of weights for interpolation

    Raises
    ------
    ValueError
        Currently supported values for `decay_method` are: "cosine", "polynomial"
    """
    if decay_method not in ["cosine", "polynomial"]:
        raise ValueError(  # noqa: TRY003
            "Currently supported values for `decay_method`",
            f"are 'cosine' and 'polynomial'. Got {decay_method=}.",
        )

    if (decay_method == "polynomial") and ("pow" not in kwargs.keys()):
        raise TypeError(  # noqa: TRY003
            "The decay_method='polynomial' expects a 'pow' argument.",
            "Please pass a 'pow' argument greater or equal to 1.",
        )

    if not np.isin(
        np.float32(timeseries_harmonisee.time_axis), np.float32(harmonisation_time)
    ).any():
        raise NotImplementedError(
            f"{harmonisation_time=} is not a value in "
            f"{timeseries_harmonisee.time_axis=}"
        )
    # initialize variable
    fill_with_zeros: npt.NDArray[Any]

    if convergence_time is None:
        time_interp = timeseries_harmonisee.time_axis[
            np.where(timeseries_harmonisee.time_axis >= harmonisation_time)
        ]
        # decay_range = len(time_axis)
        fill_with_zeros = np.array([])

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
        weight_seq = cosine_decay(len(time_interp))
    elif decay_method == "polynomial":
        # extract required additional argument
        pow: Union[float, int] = kwargs["pow"]
        weight_seq = polynomial_decay(len(time_interp), pow=pow)

    # compute weight
    weight_sequence: npt.NDArray[Any] = np.concatenate((weight_seq, fill_with_zeros))

    return weight_sequence


def interpolate_timeseries(
    harmonisee: Spline,
    harmonised: Spline,
    harmonisation_time: Union[int, float],
    timeseries_harmonisee: Timeseries,
    decay_weights: npt.NDArray[Any],
) -> Timeseries:
    """
    Compute interpolated timeseries

    The interpolated timeseries is generated by interpolating
    between the harmonised spline at harmonisation time
    and the target spline at either
    the last date of the harmonisee or the specified convergence time.

    Parameters
    ----------
    harmonisee
        harmonisee spline

    harmonised
        harmonised (adjusted) spline

    harmonisation_time
        time point at which harmonisee and target should match

    timeseries_harmonisee
        timeseries of the harmonisee

    decay_weights
        sequence of weights decaying from 1 to 0

    Returns
    -------
    timeseries_interpolated :
        timeseries that interpolate between harmonised spline and harmonisee
    """
    if not np.isin(
        np.float32(timeseries_harmonisee.time_axis), np.float32(harmonisation_time)
    ).any():
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


def interpolate_harmoniser(  # noqa: PLR0913
    interpolation_target: Spline,
    harmonised_spline: Spline,
    harmonisee_timeseries: Timeseries,
    convergence_time: Optional[Union[int, float]],
    harmonisation_time: Union[int, float],
    decay_method: str = "cosine",
    **kwargs: Any,
) -> Timeseries:
    """
    Compute an interpolated timeseries

    The interpolated timeseries is generated by interpolating
    from the harmonised_spline to the interpolation target.

    Parameters
    ----------
    interpolation_target
        interpolation target, i.e., the target
        with which predicitons the interpolation spline match after the convergence
        time?
        Usually this will be either the original harmonisee
        or the biased-corrected harmonisee

    harmonised_spline
        harmonised spline that matches with target wrt zero-and first-order derivative

    harmonisee_timeseries
        harmonisee timeseries

    convergence_time
        time point where interpolation_target and harmonised spline should match

    harmonisation_time
        time point where harmonised spline should match the original target

    decay_method
        decay method used for computing weights
        that interpolate the spline, currently supported methods are 'cosine'.

    Returns
    -------
    interpolated_timeseries :
        interpolated values
    """
    # get interpolation weights
    weights = decay_weights(
        harmonisee_timeseries,
        convergence_time=convergence_time,
        harmonisation_time=harmonisation_time,
        decay_method=decay_method,
        **kwargs,
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
    target: Spline,
    harmonisee: Spline,
    harmonisation_time: Union[int, float],
    **kwargs: Any,
) -> Spline:
    """
    Harmonises two splines by matching a harmonisee to a target spline

    Parameters
    ----------
    target
        target spline

    harmonisee
        harmonisee spline

    harmonisation_time
        time point at which harmonisee should be matched to the target

    **kwargs
        keyword arguments passed to make_interp_spline or polynomial_decay function

    Returns
    -------
    harmonised_spline :
        harmonised spline (harmonised spline
        and target have same zero-and first-order derivative at harmonisation time)
    """
    # compute derivatives
    target_dspline = target.derivative()
    harmonisee_dspline = harmonisee.derivative()

    # match first-order derivatives
    harmonised_first_derivative = harmonise_constant_offset(
        target=target_dspline,
        harmonisee=harmonisee_dspline,
        harmonisation_time=harmonisation_time,
    )

    # integrate to match zero-order derivative
    harmonised_spline_first_derivative_only = (
        harmonised_first_derivative.antiderivative()
    )

    # match zero-order derivatives
    harmonised_spline = harmonise_constant_offset(
        target=target,
        harmonisee=harmonised_spline_first_derivative_only,
        harmonisation_time=harmonisation_time,
    )

    return harmonised_spline
