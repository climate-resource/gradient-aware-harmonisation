"""
Implements the decay/convergence methods
"""

from typing import Any, Optional, Union, overload

import numpy as np
import numpy.typing as npt
from attrs import define

from gradient_aware_harmonisation.spline import (
    NP_ARRAY_OF_FLOAT_OR_INT,
    NP_FLOAT_OR_INT,
    ProductOfSplines,
    Spline,
    SumOfSplines,
)
from gradient_aware_harmonisation.timeseries import Timeseries


@define
class CosineDecaySplineHelper:
    """
    Spline that supports being used as a cosine-decay between splines

    Between `initial_time` and `final_time`,
    we return values based on a cosine-decay between 1 and 0
    if `self.apply_to_convergence` is `False`,
    otherwise we return values based on a cosine-increase between 0 and 1.
    """

    initial_time: Union[float, int]
    """
    At and before this time, we return values from `initial`
    """

    final_time: Union[float, int]
    """
    At and after this time, we return values from `final`
    """

    apply_to_convergence: bool = False
    """
    Is this helper being applied to the convergence spline?

    If `True`, we return 1 - the weights, rather than the weights.
    """

    # domain: ClassVar[list[float, float]] = [
    #     np.finfo(np.float64).tiny,
    #     np.finfo(np.float64).max,
    # ]
    # """Domain of spline"""

    @overload
    def __call__(self, x: int | float) -> int | float: ...

    @overload
    def __call__(self, x: NP_FLOAT_OR_INT) -> NP_FLOAT_OR_INT: ...

    @overload
    def __call__(self, x: NP_ARRAY_OF_FLOAT_OR_INT) -> NP_ARRAY_OF_FLOAT_OR_INT: ...

    def __call__(
        self, x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT
    ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
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

        def decay(x):
            """Get cosine-decay"""
            angle = (
                np.pi * (x - self.initial_time) / (self.final_time - self.initial_time)
            )
            gamma = 0.5 * (1 + np.cos(angle))

            if self.apply_to_convergence:
                return 1 - gamma

            return gamma

        if not isinstance(x, np.ndarray):
            if x <= self.initial_time:
                if self.apply_to_convergence:
                    return 0.0
                return 1.0

            if x >= self.final_time:
                if self.apply_to_convergence:
                    return 1.0
                return 0.0

            return decay(x)

        x_gte_final_time = np.where(x >= self.final_time)
        x_decay = np.logical_and(np.where(x > self.initial_time), ~x_gte_final_time)
        gamma = np.ones_like(x)
        gamma[x_gte_final_time] = 0.0
        gamma[x_decay] = decay(x[x_decay])

        if self.apply_to_convergence:
            return 1 - gamma

        return gamma

    def derivative(self) -> SumOfSplines:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        return CosineDecaySplineHelperDerivative(
            initial_time=self.initial_time,
            final_time=self.final_time,
            apply_to_convergence=self.apply_to_convergence,
        )

    def antiderivative(self) -> SumOfSplines:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        raise NotImplementedError


@define
class CosineDecaySplineHelperDerivative:
    """
    Derivative of [CosineDecaySplineHelper][(m).]
    """

    initial_time: Union[float, int]
    """
    Initial time of the cosine-decay
    """

    final_time: Union[float, int]
    """
    Final time of the cosine-decay
    """

    apply_to_convergence: bool = False
    """
    Is this helper being applied to the convergence spline?
    """

    # domain: ClassVar[list[float, float]] = [
    #     np.finfo(np.float64).tiny,
    #     np.finfo(np.float64).max,
    # ]
    # """Domain of spline"""

    @overload
    def __call__(self, x: int | float) -> int | float: ...

    @overload
    def __call__(self, x: NP_FLOAT_OR_INT) -> NP_FLOAT_OR_INT: ...

    @overload
    def __call__(self, x: NP_ARRAY_OF_FLOAT_OR_INT) -> NP_ARRAY_OF_FLOAT_OR_INT: ...

    def __call__(
        self, x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT
    ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
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

        def decay_derivative(x):
            """Get cosine-decay derivative"""
            angle = (
                np.pi * (x - self.initial_time) / (self.final_time - self.initial_time)
            )
            gamma_derivative = -0.5 * np.sin(angle)

            if self.apply_to_convergence:
                return -gamma_derivative

            return gamma_derivative

        if not isinstance(x, np.ndarray):
            if x <= self.initial_time:
                return 0.0

            if x >= self.final_time:
                return 0.0

            return decay_derivative(x)

        x_decay = np.logical_and(
            np.where(x > self.initial_time), np.where(x < self.final_time)
        )
        gamma = np.zeros_like(x)
        gamma[x_decay] = decay_derivative(x[x_decay])

        return gamma

    def derivative(self) -> SumOfSplines:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        return CosineDecaySplineHelperDerivative(
            initial_time=self.initial_time,
            final_time=self.final_time,
            apply_to_convergence=self.apply_to_convergence,
        )

    def antiderivative(self) -> SumOfSplines:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        raise NotImplementedError


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


def get_cosine_decay_harmonised_spline(
    harmonisation_time: Union[int, float],
    convergence_time: Union[int, float],
    harmonised_spline_no_convergence: Spline,
    convergence_spline: Spline,
) -> ProductOfSplines:
    """
    Generate the harmonised spline

    Parameters
    ----------
    harmonisation_time
        Harmonisation time

        This is the time at and before which
        the solution should be equal to `harmonised_spline_no_convergence`.

    convergence_time
        Convergence time

        This is the time at and after which
        the solution should be equal to `convergence_spline`.

    harmonised_spline_no_convergence
        Harmonised spline that does not consider convergence

    convergence_spline
        The spline to which the result should converge

    Returns
    -------
    :
        Harmonised spline
    """
    return SumOfSplines(
        ProductOfSplines(
            CosineDecaySplineHelper(
                initial_time=harmonisation_time, final_time=convergence_time
            ),
            harmonised_spline_no_convergence,
        ),
        ProductOfSplines(
            CosineDecaySplineHelper(
                initial_time=harmonisation_time,
                final_time=convergence_time,
                apply_to_convergence=True,
            ),
            convergence_spline,
        ),
    )


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
