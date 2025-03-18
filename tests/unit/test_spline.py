"""
Tests of `gradient_aware_harmonisation.spline`
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.interpolate

from gradient_aware_harmonisation.spline import (
    SplineScipy,
    add_constant_to_spline,
)

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

    np.testing.assert_equal(start_derivative(x_fine) + const, res_derivative(x_fine))


@pytest.mark.parametrize("const", (-1.3, 0.0, 2.5))
def test_add_constant_to_spline_antiderivative(const):
    pass
