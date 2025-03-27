"""
Implements the decay/convergence methods
"""

from __future__ import annotations

from typing import Union, overload

import numpy as np
from attrs import define

from gradient_aware_harmonisation.spline import (
    NP_ARRAY_OF_FLOAT_OR_INT,
    NP_FLOAT_OR_INT,
    ProductOfSplines,
    Spline,
    SumOfSplines,
)


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

        def decay_derivative(
            x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT,
        ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
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

    def derivative(self) -> CosineDecaySplineHelperDerivative:
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

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


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

    def __call__(  # noqa: PLR0911
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

        def decay(
            x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT,
        ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
            """Get cosine-decay"""
            angle: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT = np.divide(
                np.pi * (x - self.initial_time), (self.final_time - self.initial_time)
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
        x_decay = np.logical_and(x > self.initial_time, x < self.final_time)
        gamma = np.ones_like(x)
        gamma[x_gte_final_time] = 0.0
        gamma[x_decay] = decay(x[x_decay])

        if self.apply_to_convergence:
            return 1 - gamma

        return gamma

    def derivative(self) -> CosineDecaySplineHelperDerivative:
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


def get_cosine_decay_harmonised_spline(
    harmonisation_time: Union[int, float],
    convergence_time: Union[int, float],
    harmonised_spline_no_convergence: Spline,
    convergence_spline: Spline,
) -> SumOfSplines:
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
