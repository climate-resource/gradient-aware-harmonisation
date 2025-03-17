"""
Utility functions
"""

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt
from attrs import define, field

from gradient_aware_harmonisation.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import scipy.interpolate
    from typing_extensions import TypeAlias

# If you want, break these out into `gradient_aware_harmonisation.typing.py`
NP_FLOAT_OR_INT: TypeAlias = Union[np.floating[Any], np.integer[Any]]
"""
Type alias for a numpy float or int (not complex)
"""

NP_ARRAY_OF_FLOAT_OR_INT: TypeAlias = npt.NDArray[NP_FLOAT_OR_INT]
"""
Type alias for an array of numpy float or int (not complex)
"""


@define
class Spline(Protocol):
    """
    Single spline
    """

    @overload
    def __call__(self, x: int | float) -> int | float: ...

    @overload
    def __call__(self, x: NP_FLOAT_OR_INT) -> NP_FLOAT_OR_INT: ...

    @overload
    def __call__(self, x: NP_ARRAY_OF_FLOAT_OR_INT) -> NP_ARRAY_OF_FLOAT_OR_INT: ...

    def __call__(
        self, x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT
    ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
        """Get the value of the spline at a particular x-value"""

    def derivative(self) -> Spline:
        """
        Calculate the derivative of self
        """

    def antiderivative(self) -> Spline:
        """
        Calculate the anti-derivative/integral of self
        """


@define
class Splines:
    """
    Spline class combined (target and harmonisee)
    """

    # target: Spline
    # harmonisee: Spline
    target: scipy.interpolate.BSpline
    harmonisee: scipy.interpolate.BSpline


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


def timeseries_to_spline(
    timeseries: Timeseries, **kwargs: Any
) -> scipy.interpolate.BSpline:
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

    spline = scipy.interpolate.make_interp_spline(
        timeseries.time_axis, timeseries.values, **kwargs_spline
    )

    return spline


def harmonise_timeseries(
    # target: Spline,
    # harmonisee: Spline,
    target: scipy.interpolate.BSpline,
    harmonisee: scipy.interpolate.BSpline,
    timeseries_harmonisee: Timeseries,
    harmonisation_time: Union[int, float],
) -> Timeseries:
    """
    Compute a timeseries based on the adjustment of the harmonisee to the target.

    Parameters
    ----------
    target
        target spline from timeseries array

    harmonisee
        harmonisee spline from timeseries array (should be adjusted to target spline)

    timeseries_harmonisee
        harmonisee timeseries of format dict(time_axis = np.array, values = np.array)

    harmonisation_time
        point in time_axis at which harmonisee should be matched to target

    Returns
    -------
    harmonised_timeseries :
        harmonised timeseries
    """
    diff = target(harmonisation_time) - harmonisee(harmonisation_time)
    harmonised_values = harmonisee(timeseries_harmonisee.time_axis) + diff

    harmonised_timeseries = Timeseries(
        time_axis=timeseries_harmonisee.time_axis,
        values=harmonised_values,
    )

    return harmonised_timeseries


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
    # harmonisee: Spline,
    # harmonised: Spline,
    harmonisee: scipy.interpolate.BSpline,
    harmonised: scipy.interpolate.BSpline,
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
    # timeseries harmonised
    # timeseries_harmonised = harmonised(timeseries_harmonisee.time_axis.values)
    # reduce timeseries from harmonisation time point

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


# %% Wrapper
def compute_splines(
    target: Timeseries, harmonisee: Timeseries, **kwargs: Any
) -> Splines:
    """
    Convert input arrays into timeseries objects and compute splines

    Parameters
    ----------
    target
        Timeseries of target data

    harmonisee
        timeseries of matching data (have to be adjusted to match the target)

    **kwargs
        keyword arguments passed to make_interp_spline

    Returns
    -------
    splines :
        splines of target and harmonisee
    """
    # compute splines
    target_spline = timeseries_to_spline(target, **kwargs)
    harmonisee_spline = timeseries_to_spline(harmonisee, **kwargs)

    splines = Splines(target=target_spline, harmonisee=harmonisee_spline)
    return splines


def interpolate_harmoniser(  # noqa: PLR0913
    interpolation_target: scipy.interpolate.BSpline,
    harmonised_spline: scipy.interpolate.BSpline,
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
    splines: Splines,
    harmonisee_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    **kwargs: Any,
) -> scipy.interpolate.BSpline:
    """
    Harmonises two splines by matching a harmonisee to a target spline

    Parameters
    ----------
    splines
        splines of target and harmonisee as computed by :func:`compute_splines`

    harmonisee_timeseries
        timeseries of matching data

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
    target_dspline = splines.target.derivative()
    harmonisee_dspline = splines.harmonisee.derivative()

    # match first-order derivatives
    harmonised_d1_timeseries = harmonise_timeseries(
        target_dspline, harmonisee_dspline, harmonisee_timeseries, harmonisation_time
    )
    # compute spline
    harmonised_D1_spline = timeseries_to_spline(harmonised_d1_timeseries, **kwargs)
    # integrate to match zero-order derivative
    harmonised_d1_spline = harmonised_D1_spline.antiderivative()

    # match zero-order derivatives
    harmonised_d0d1_timeseries = harmonise_timeseries(
        splines.target,
        harmonised_d1_spline,
        harmonisee_timeseries,
        harmonisation_time,
    )
    # compute spline
    harmonised_d0d1_spline = timeseries_to_spline(harmonised_d0d1_timeseries, **kwargs)

    return harmonised_d0d1_spline


def biased_corrected_harmonisee(
    splines: Splines,
    harmonisee_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    **kwargs: Any,
) -> Any:
    """
    Compute the biased corrected spline

    This is the harmonisee matches the target spline wrt the zero-order
    derivative.

    Parameters
    ----------
    splines
        splines of target and harmonisee as computed by :func:`compute_splines`

    harmonisee_timeseries
        timeseries of matching data

    harmonisation_time
        time point at which harmonisee should be matched to the target

    **kwargs
        keyword arguments passed to make_interp_spline or polynomial_decay function

    Returns
    -------
    biased_corrected_spline :
        biased corrected spline
    """
    biased_corrected_timeseries = harmonise_timeseries(
        splines.target,
        splines.harmonisee,
        harmonisee_timeseries,
        harmonisation_time,
    )
    biased_corrected_spline = timeseries_to_spline(
        biased_corrected_timeseries, **kwargs
    )

    return biased_corrected_spline
