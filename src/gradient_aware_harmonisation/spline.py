"""
Spline handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Union, overload

import numpy as np
import numpy.typing as npt
from attrs import define

if TYPE_CHECKING:
    import scipy.interpolate
    from typing_extensions import TypeAlias

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
class SplineScipyBSpline:
    """
    An adapter which wraps an instance of [scipy.interpolate.BSpline][]
    """

    scipy_spline: scipy.interpolate.BSpline

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
        return self.scipy_spline(x)

    def derivative(self) -> SplineScipyBSpline:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        return SplineScipyBSpline(self.scipy_spline.derivative())

    def antiderivative(self) -> SplineScipyBSpline:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        return SplineScipyBSpline(self.scipy_spline.antiderivative())


@define
class SplinePlusConstant:
    """
    A spline plus a constant
    """

    spline: Spline
    """Spline"""

    constant: float | int
    """Constant to add to the `spline`"""

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
        return self.spline(x) + self.constant

    def derivative(self) -> SplineScipyBSpline:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        return self.derivative()

    def antiderivative(self) -> SplineScipyBSpline:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        raise NotImplementedError


def add_constant_to_spline(in_spline: Spline, constant: float | int) -> Spline:
    """
    Add a constant value to a spline

    Parameters
    ----------
    in_spline
        Input spline

    constant
        Constant to add

    Returns
    -------
    :
        Spline plus the given constant
    """
    return SplinePlusConstant(spline=in_spline, constant=constant)
