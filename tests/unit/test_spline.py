"""
Tests of `gradient_aware_harmonisation.spline`
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.interpolate

from gradient_aware_harmonisation.spline import (
    SplineScipyBSpline,
    add_constant_to_spline,
)

# TODO:
# - tests of SplineScipyBSpline


@pytest.mark.parametrize("const", (-1.3, 0.0, 2.5))
def test_add_constant_to_spline(const):
    x_values = np.array([0.0, 1.0, 2.0, 3.0])
    start = SplineScipyBSpline(
        scipy.interpolate.make_interp_spline(x_values, np.array([1.0, 2.0, 3.0, 4.0]))
    )

    res = add_constant_to_spline(start, const)

    x_fine = np.linspace(x_values.min(), x_values.max(), 101)

    np.testing.assert_equal(start(x_fine) + const, res(x_fine))
    # TODO:
    # - test derivative
    # - test antiderivative
