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


class Spline(Protocol):
    """
    Single spline
    """

    # domain: [float, float]
    # """Domain over the spline can be used"""

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
class SplineScipy:
    """
    An adapter which wraps various classes from [scipy.interpolate][]
    """

    # domain: ClassVar[list[float, float]] = [
    #     np.finfo(np.float64).tiny,
    #     np.finfo(np.float64).max,
    # ]
    # """domain of spline (reals)"""

    scipy_spline: scipy.interpolate.BSpline | scipy.interpolate.PPoly

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
        return self.scipy_spline(x)

    def derivative(self) -> SplineScipy:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        return SplineScipy(self.scipy_spline.derivative())

    def antiderivative(self) -> SplineScipy:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        return SplineScipy(self.scipy_spline.antiderivative())


@define
class SumOfSplines:
    """
    Sum of two splines
    """

    spline_one: Spline
    """First spline"""

    spline_two: Spline
    """Second spline"""

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
        return self.spline_one(x) + self.spline_two(x)

    def derivative(self) -> SumOfSplines:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        return SumOfSplines(self.spline_one.derivative(), self.spline_two.derivative())

    def antiderivative(self) -> SumOfSplines:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        return SumOfSplines(
            self.spline_one.antiderivative(), self.spline_two.antiderivative()
        )


@define
class ProductOfSplines:
    """
    Product of two splines
    """

    spline_one: Spline
    """First spline"""

    spline_two: Spline
    """Second spline"""

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
        return self.spline_one(x) * self.spline_two(x)

    def derivative(self) -> SumOfSplines:
        """
        Calculate the derivative of self

        Returns
        -------
        :
            Derivative of self
        """
        # use the product rule in order to get the derivative of the product
        # of two splines
        return SumOfSplines(
            ProductOfSplines(self.spline_one, self.spline_two.derivative()),
            ProductOfSplines(self.spline_one.derivative(), self.spline_two),
        )

    def antiderivative(self) -> SumOfSplines:
        """
        Calculate the anti-derivative/integral of self

        Returns
        -------
        :
            Anti-derivative of self
        """
        # computation of the antiderivative of a product of splines is not
        # straightforward
        # However, as we don't need the antiderivative in our current workflow
        # we leave it for the time being as "not implemented"
        raise NotImplementedError
