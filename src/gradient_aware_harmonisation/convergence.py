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

        def calc_gamma(
            x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT,
        ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
            """Get cosine-decay derivative"""
            # compute weight (here: gamma) according to a cosine-decay
            angle = (
                np.pi * (x - self.initial_time) / (self.final_time - self.initial_time)
            )

            gamma_decaying = 0.5 * (1 + np.cos(angle))

            return gamma_decaying

        if not isinstance(x, np.ndarray):
            if x <= self.initial_time:
                gamma: float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT = 1.0
            elif x >= self.final_time:
                gamma = 0.0
            else:
                gamma = calc_gamma(x)

            # The weighted sum for computing the harmonised AND converged
            # function has the form: "gamma * harmonised + (1-gamma) * convergence".
            # Depending on which product we want to compute (LHS or RHS of sum),
            # we need gamma or 1-gamma, therefore we include the following condition
            # in all our return statements.
            if self.apply_to_convergence:
                return 1.0 - gamma

            return gamma

        # apply decay function only to values that lie between harmonisation
        # time and convergence-time
        x_gte_final_time = np.where(x >= self.final_time)
        x_decay = np.logical_and(x >= self.initial_time, x < self.final_time)
        gamma = np.ones_like(x, dtype=np.floating)
        gamma[x_gte_final_time] = 0.0
        gamma[x_decay] = calc_gamma(x[x_decay])

        if self.apply_to_convergence:
            return 1.0 - gamma

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

    def antiderivative(self) -> CosineDecaySplineHelperDerivative:
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

        # compute weight (here: gamma) according to a cosine-decay
        def calc_gamma_rising_derivative(
            x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT,
        ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
            """Get cosine-decay derivative"""
            # compute derivative of gamma according to a cosine-decay
            angle = (
                np.pi * (x - self.initial_time) / (self.final_time - self.initial_time)
            )

            const_factor = -np.pi / (2 * (self.final_time - self.initial_time))
            gamma_decaying_derivative = const_factor * np.sin(angle)

            return gamma_decaying_derivative

        if not isinstance(x, np.ndarray):
            if x <= self.initial_time or x >= self.final_time:
                return 0.0

            gamma_rising_derivative = calc_gamma_rising_derivative(x)

            # The weighted sum for computing the harmonised AND converged
            # function has the form: "gamma * harmonised + (1-gamma) * convergence".
            # Depending on which product we want to compute (LHS or RHS of sum),
            # we need gamma or 1-gamma, therefore we include the following condition
            # in all our return statements.
            if self.apply_to_convergence:
                return -gamma_rising_derivative

            return gamma_rising_derivative

        # apply decay function only to values that lie between harmonisation
        # time and convergence-time
        x_decay = np.where(np.logical_and(x > self.initial_time, x < self.final_time))
        gamma_rising_derivative = np.zeros_like(x, dtype=np.floating)
        gamma_rising_derivative[x_decay] = calc_gamma_rising_derivative(x[x_decay])

        if self.apply_to_convergence:
            return -gamma_rising_derivative
        return gamma_rising_derivative

    def derivative(self) -> CosineDecaySplineHelperDerivative:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        raise NotImplementedError

    def antiderivative(self) -> CosineDecaySplineHelperDerivative:
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


def get_cosine_decay_harmonised_spline(
    harmonisation_time: Union[int, float],
    convergence_time: Union[int, float],
    harmonised_spline_no_convergence: Spline,
    convergence_spline: Spline,
) -> SumOfSplines:
    """
    Generate the harmonised spline based on a cosine-decay

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
    # The harmonised spline is considered as the spline that match
    # the target-spline at the harmonisation time (wrt to zero-and
    # first order derivative). Then we use a decay function to let
    # the harmonised spline converge to the convergence-spline.
    # This decay function has the form of a weighted sum:
    # weight * harmonised_spline + (1-weight) * convergence_spline
    # With weights decaying from 1 to 0 whereby the decay trajectory
    # is determined by the cosine decay.
    return SumOfSplines(
        ProductOfSplines(
            CosineDecaySplineHelper(
                initial_time=harmonisation_time,
                final_time=convergence_time,
                apply_to_convergence=False,
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


@define
class PolynomialDecaySplineHelper:
    """
    Spline that supports being used as a polynomial-decay between splines

    Between `initial_time` and `final_time`,
    we return values based on a polynomial-decay between 1 and 0
    if `self.apply_to_convergence` is `False`,
    otherwise we return values based on a polynomial-increase between 0 and 1.
    """

    initial_time: Union[float, int]
    """
    At and before this time, we return values from `initial`
    """

    final_time: Union[float, int]
    """
    At and after this time, we return values from `final`
    """

    power: Union[float, int]
    """
    Power of polynomial decay; with power >= 1
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

        def calc_gamma(
            x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT,
        ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
            """Get polynomial-decay"""
            # compute weight (here: gamma) according to polynomial-decay
            gamma_decaying = (
                1 - (x - self.initial_time) / (self.final_time - self.initial_time)
            ) ** self.power

            return gamma_decaying

        if not isinstance(x, np.ndarray):
            if x <= self.initial_time:
                gamma: float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT = 1.0
            elif x >= self.final_time:
                gamma = 0.0
            else:
                gamma = calc_gamma(x)

            # The weighted sum for computing the harmonised AND converged
            # function has the form: "gamma * harmonised + (1-gamma) * convergence".
            # Depending on which product we want to compute (LHS or RHS of sum),
            # we need gamma or 1-gamma, therefore we include the following condition
            # in all our return statements.
            if self.apply_to_convergence:
                return 1.0 - gamma

            return gamma

        # apply decay function only to values that lie between harmonisation
        # time and convergence-time
        x_gte_final_time = np.where(x >= self.final_time)
        x_decay = np.logical_and(x >= self.initial_time, x < self.final_time)
        gamma = np.ones_like(x, dtype=np.floating)
        gamma[x_gte_final_time] = 0.0
        gamma[x_decay] = calc_gamma(x[x_decay])

        if self.apply_to_convergence:
            return 1.0 - gamma

        return gamma

    def derivative(self) -> PolynomialDecaySplineHelperDerivative:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self

        """
        return PolynomialDecaySplineHelperDerivative(
            initial_time=self.initial_time,
            final_time=self.final_time,
            power=self.power,
            apply_to_convergence=self.apply_to_convergence,
        )

    def antiderivative(self) -> PolynomialDecaySplineHelperDerivative:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        raise NotImplementedError


@define
class PolynomialDecaySplineHelperDerivative:
    """
    Derivative of [PolynomialDecaySplineHelper][(m).]
    """

    initial_time: Union[float, int]
    """
    Initial time of the polynomial-decay
    """

    final_time: Union[float, int]
    """
    Final time of the polynomial-decay
    """

    power: Union[float, int]
    """
    Power of polynomial decay; with power >= 1
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

        # compute weight (here: gamma) according to a polynomial-decay
        def calc_gamma_rising_derivative(
            x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT,
        ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
            """Get polynomial-decay derivative"""
            # compute derivative of gamma according to a polynomial-decay
            numerator = (
                self.power
                * (1 - (x - self.initial_time) / (self.final_time - self.initial_time))
                ** self.power
            )
            denominator = x - self.final_time

            gamma_decaying_derivative = numerator / denominator

            return gamma_decaying_derivative

        if not isinstance(x, np.ndarray):
            if x <= self.initial_time or x >= self.final_time:
                return 0.0

            gamma_rising_derivative = calc_gamma_rising_derivative(x)

            # The weighted sum for computing the harmonised AND converged
            # function has the form: "gamma * harmonised + (1-gamma) * convergence".
            # Depending on which product we want to compute (LHS or RHS of sum),
            # we need gamma or 1-gamma, therefore we include the following condition
            # in all our return statements.
            if self.apply_to_convergence:
                return -gamma_rising_derivative

            return gamma_rising_derivative

        # apply decay function only to values that lie between harmonisation
        # time and convergence-time
        x_decay = np.where(np.logical_and(x > self.initial_time, x < self.final_time))
        gamma_rising_derivative = np.zeros_like(x, dtype=np.floating)
        gamma_rising_derivative[x_decay] = calc_gamma_rising_derivative(x[x_decay])

        if self.apply_to_convergence:
            return -gamma_rising_derivative
        return gamma_rising_derivative

    def derivative(self) -> PolynomialDecaySplineHelperDerivative:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        raise NotImplementedError

    def antiderivative(self) -> PolynomialDecaySplineHelperDerivative:
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


def get_polynomial_decay_harmonised_spline(
    harmonisation_time: Union[int, float],
    convergence_time: Union[int, float],
    harmonised_spline_no_convergence: Spline,
    convergence_spline: Spline,
    power: Union[int, float],
) -> SumOfSplines:
    """
    Generate the harmonised spline based on a polynomial-decay

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

    power
        The power of the polynomial-decay, whereby power >= 1.
        With power = 1 being equivalent to linear decay.

    Returns
    -------
    :
        Harmonised spline
    """
    # The harmonised spline is considered as the spline that match
    # the target-spline at the harmonisation time (wrt to zero-and
    # first order derivative). Then we use a decay function to let
    # the harmonised spline converge to the convergence-spline.
    # This decay function has the form of a weighted sum:
    # weight * harmonised_spline + (1-weight) * convergence_spline
    # With weights decaying from 1 to 0 whereby the decay trajectory
    # is determined by the polynomial decay.
    return SumOfSplines(
        ProductOfSplines(
            PolynomialDecaySplineHelper(
                initial_time=harmonisation_time,
                final_time=convergence_time,
                power=power,
                apply_to_convergence=False,
            ),
            harmonised_spline_no_convergence,
        ),
        ProductOfSplines(
            PolynomialDecaySplineHelper(
                initial_time=harmonisation_time,
                final_time=convergence_time,
                power=power,
                apply_to_convergence=True,
            ),
            convergence_spline,
        ),
    )


@define
class LogisticDecaySplineHelper:
    """
    Spline that supports being used as a logistic-decay between splines

    Between `initial_time` and `final_time`,
    we return values based on a polynomial-decay between 1 and 0
    if `self.apply_to_convergence` is `False`,
    otherwise we return values based on a logistic-increase between 0 and 1.
    """

    initial_time: Union[float, int]
    """
    At and before this time, we return values from `initial`
    """

    final_time: Union[float, int]
    """
    At and after this time, we return values from `final`
    """

    slope: Union[float, int]
    """
    Slope of logistic function
    """

    shift: Union[float, int]
    """
    Shift of logistic function
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

        def calc_gamma(
            x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT,
        ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
            """Get polynomial-decay"""
            # compute weight (here: gamma) according to logistic-decay

            x_prime = x - self.initial_time
            delta = self.final_time - self.initial_time

            gamma_decaying = 1 - 1 / (
                1
                + np.exp(
                    -2 * np.exp(self.slope) * delta * x_prime + 3 * delta + self.shift
                )
            )

            return gamma_decaying

        if not isinstance(x, np.ndarray):
            if x <= self.initial_time:
                gamma: float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT = 1.0
            elif x >= self.final_time:
                gamma = 0.0
            else:
                gamma = calc_gamma(x)

            # The weighted sum for computing the harmonised AND converged
            # function has the form: "gamma * harmonised + (1-gamma) * convergence".
            # Depending on which product we want to compute (LHS or RHS of sum),
            # we need gamma or 1-gamma, therefore we include the following condition
            # in all our return statements.
            if self.apply_to_convergence:
                return 1.0 - gamma

            return gamma

        # apply decay function only to values that lie between harmonisation
        # time and convergence-time
        x_gte_final_time = np.where(x >= self.final_time)
        x_decay = np.logical_and(x >= self.initial_time, x < self.final_time)
        gamma = np.ones_like(x, dtype=np.floating)
        gamma[x_gte_final_time] = 0.0
        gamma[x_decay] = calc_gamma(x[x_decay])

        if self.apply_to_convergence:
            return 1.0 - gamma

        return gamma

    def derivative(self) -> LogisticDecaySplineHelperDerivative:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self

        """
        return LogisticDecaySplineHelperDerivative(
            initial_time=self.initial_time,
            final_time=self.final_time,
            slope=self.slope,
            shift=self.shift,
            apply_to_convergence=self.apply_to_convergence,
        )

    def antiderivative(self) -> LogisticDecaySplineHelperDerivative:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        raise NotImplementedError


@define
class LogisticDecaySplineHelperDerivative:
    """
    Derivative of [LogisticDecaySplineHelper][(m).]
    """

    initial_time: Union[float, int]
    """
    Initial time of the logistic-decay
    """

    final_time: Union[float, int]
    """
    Final time of the logistic-decay
    """

    slope: Union[float, int]
    """
    Slope of logistic function
    """

    shift: Union[float, int]
    """
    Shift of logistic function
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

        # compute weight (here: gamma) according to a polynomial-decay
        def calc_gamma_rising_derivative(
            x: int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT,
        ) -> int | float | NP_FLOAT_OR_INT | NP_ARRAY_OF_FLOAT_OR_INT:
            """Get logistic-decay derivative"""
            # compute derivative of gamma according to a logistic-decay
            x_prime = x - self.initial_time
            delta = self.final_time - self.initial_time
            scaler = 2 * self.slope * delta

            numerator = scaler * np.exp(-scaler * x_prime + 3 * delta + self.shift)
            denominator = np.exp(-scaler * x_prime + 3 * delta + self.shift)

            gamma_decaying_derivative = -numerator / (denominator + 1) ** 2

            return gamma_decaying_derivative

        if not isinstance(x, np.ndarray):
            if x <= self.initial_time or x >= self.final_time:
                return 0.0

            gamma_rising_derivative = calc_gamma_rising_derivative(x)

            # The weighted sum for computing the harmonised AND converged
            # function has the form: "gamma * harmonised + (1-gamma) * convergence".
            # Depending on which product we want to compute (LHS or RHS of sum),
            # we need gamma or 1-gamma, therefore we include the following condition
            # in all our return statements.
            if self.apply_to_convergence:
                return -gamma_rising_derivative

            return gamma_rising_derivative

        # apply decay function only to values that lie between harmonisation
        # time and convergence-time
        x_decay = np.where(np.logical_and(x > self.initial_time, x < self.final_time))
        gamma_rising_derivative = np.zeros_like(x, dtype=np.floating)
        gamma_rising_derivative[x_decay] = calc_gamma_rising_derivative(x[x_decay])

        if self.apply_to_convergence:
            return -gamma_rising_derivative
        return gamma_rising_derivative

    def derivative(self) -> LogisticDecaySplineHelperDerivative:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        raise NotImplementedError

    def antiderivative(self) -> LogisticDecaySplineHelperDerivative:
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


def get_logistic_decay_harmonised_spline(  # noqa: PLR0913
    harmonisation_time: Union[int, float],
    convergence_time: Union[int, float],
    harmonised_spline_no_convergence: Spline,
    convergence_spline: Spline,
    slope: Union[int, float],
    shift: Union[int, float],
) -> SumOfSplines:
    """
    Generate the harmonised spline based on a logistic-decay

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

    slope
        Determine steepness of logistic function
        Slope < 0: decrease slope; slope > 0: increase slope

    shift
        Shift the logistic function horizontally
        Shift < 0: left and shift > 0: right

    Returns
    -------
    :
        Harmonised spline
    """
    # The harmonised spline is considered as the spline that match
    # the target-spline at the harmonisation time (wrt to zero-and
    # first order derivative). Then we use a decay function to let
    # the harmonised spline converge to the convergence-spline.
    # This decay function has the form of a weighted sum:
    # weight * harmonised_spline + (1-weight) * convergence_spline
    # With weights decaying from 1 to 0 whereby the decay trajectory
    # is determined by the logistic decay.
    return SumOfSplines(
        ProductOfSplines(
            LogisticDecaySplineHelper(
                initial_time=harmonisation_time,
                final_time=convergence_time,
                slope=slope,
                shift=shift,
                apply_to_convergence=False,
            ),
            harmonised_spline_no_convergence,
        ),
        ProductOfSplines(
            LogisticDecaySplineHelper(
                initial_time=harmonisation_time,
                final_time=convergence_time,
                slope=slope,
                shift=shift,
                apply_to_convergence=True,
            ),
            convergence_spline,
        ),
    )
