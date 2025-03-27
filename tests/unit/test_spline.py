"""
Tests of `gradient_aware_harmonisation.spline`
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_aware_harmonisation.spline import (
    ProductOfSplines,
    SplineScipy,
)
from gradient_aware_harmonisation.utils import add_constant_to_spline

scipy = pytest.importorskip("scipy")

# TODO:
# - tests of SplineScipyBSpline


@pytest.mark.parametrize("const", (-1.3, 0.0, 2.5))
def test_add_constant_to_spline(const):
    x_values = np.array([0.0, 1.0, 2.0, 3.0])
    start = SplineScipy(
        scipy.interpolate.make_interp_spline(x_values, np.array([1.0, 2.0, 3.0, 4.0]))
    )

    res = add_constant_to_spline(start, const)

    x_fine = np.linspace(x_values.min(), x_values.max(), 101)

    np.testing.assert_equal(start(x_fine) + const, res(x_fine))
    # TODO:
    # - test derivative
    # - test antiderivative


@pytest.mark.parametrize("const", (-1.3, 0.0, 2.5))
def test_add_constant_to_spline_derivative(const):
    x_values = np.array([0.0, 1.0, 2.0, 3.0])
    start = SplineScipy(
        scipy.interpolate.make_interp_spline(x_values, np.array([1.0, 2.0, 3.0, 4.0]))
    )
    # compute first-order derivative
    start_derivative = start.derivative()
    # add constant to first-order derivative
    res_derivative = add_constant_to_spline(start, const).derivative()

    x_fine = np.linspace(x_values.min(), x_values.max(), 101)

    np.testing.assert_equal(start_derivative(x_fine), res_derivative(x_fine))


@pytest.mark.parametrize("const", (-1.3, 0.0, 2.5))
def test_add_constant_to_spline_antiderivative(const):
    pass


def test_product_of_splines():
    x_min = -10
    x_max = 10
    x_fine = np.linspace(x_min, x_max, 101)

    x_2 = SplineScipy(
        scipy.interpolate.PPoly(
            x=[x_min, x_max],
            c=[[1], [0], [0]],  # y=x^2
        )
    )
    x = SplineScipy(
        scipy.interpolate.PPoly(
            x=[x_min, x_max],
            c=[[1], [0]],  # y=x
        )
    )

    x_3 = SplineScipy(
        scipy.interpolate.PPoly(
            x=[x_min, x_max],
            c=[[1], [0], [0], [0]],  # y=x^3
        )
    )

    product = ProductOfSplines(x_2, x)

    # Make sure that starting product actually works
    np.testing.assert_equal(x_3(x_fine), product(x_fine))

    # Check derivative
    np.testing.assert_equal(x_3.derivative()(x_fine), product.derivative()(x_fine))

    # Doing antiderivative well requires basically reimplementing sympy,
    # hence not implemented.
    with pytest.raises(NotImplementedError):
        np.testing.assert_equal(
            x_3.antiderivative()(x_fine), product.antiderivative()(x_fine)
        )
