"""
Tests of `add_constant_to_spline`
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_aware_harmonisation.timeseries import Timeseries
from gradient_aware_harmonisation.utils import add_constant_to_spline

scipy = pytest.importorskip("scipy")


@pytest.mark.parametrize("harmonisation_time", [3.0, 3.5, 4.0])
def test_constant_to_spline(harmonisation_time):
    x1 = np.arange(-2, 4.1, 0.1)
    y1 = -16 * x1

    x2 = np.arange(2, 10, 0.1)
    y2 = 0.5 * x2 + x2**3

    target_timeseries = Timeseries(time_axis=x1, values=y1)
    harmonisee_timeseries = Timeseries(time_axis=x2, values=y2)

    # convert timeseries to splines
    target_spline = target_timeseries.to_spline()
    harmonisee_spline = harmonisee_timeseries.to_spline()

    # compute derivatives of splines
    target_dspline = target_spline.derivative()
    harmonisee_dspline = harmonisee_spline.derivative()

    # match first-order derivatives
    diff_dspline = np.subtract(
        target_dspline(harmonisation_time), harmonisee_dspline(harmonisation_time)
    )

    harmonised_first_derivative = add_constant_to_spline(
        in_spline=harmonisee_dspline, constant=diff_dspline
    )

    np.testing.assert_allclose(
        harmonised_first_derivative(harmonisation_time),
        target_dspline(harmonisation_time),
    )

    # integrate to match zero-order derivative
    harmonised_spline_first_derivative_only = (
        harmonised_first_derivative.antiderivative()
    )

    # match zero-order derivatives
    diff_spline = np.subtract(
        target_spline(harmonisation_time),
        harmonised_spline_first_derivative_only(harmonisation_time),
    )

    harmonised_spline_no_convergence = add_constant_to_spline(
        in_spline=harmonised_spline_first_derivative_only, constant=diff_spline
    )

    np.testing.assert_allclose(
        harmonised_spline_no_convergence(harmonisation_time),
        target_spline(harmonisation_time),
    )

    harmonised_dspline = harmonised_spline_no_convergence.derivative()

    np.testing.assert_allclose(
        harmonised_dspline(harmonisation_time), target_dspline(harmonisation_time)
    )


@pytest.mark.parametrize("harmonisation_time", [2015, 2016, 2017])
def test_constant_to_spline_real_data(harmonisation_time):
    target_timeseries = Timeseries(
        time_axis=np.arange(2015, 2100), values=np.arange(2100 - 2015)
    )
    harmonisee_timeseries = target_timeseries

    # convert timeseries to splines
    target_spline = target_timeseries.to_spline()
    harmonisee_spline = harmonisee_timeseries.to_spline()

    # compute derivatives of splines
    target_dspline = target_spline.derivative()
    harmonisee_dspline = harmonisee_spline.derivative()

    # match first-order derivatives
    diff_dspline = np.subtract(
        target_dspline(harmonisation_time), harmonisee_dspline(harmonisation_time)
    )

    harmonised_first_derivative = add_constant_to_spline(
        in_spline=harmonisee_dspline, constant=diff_dspline
    )

    np.testing.assert_allclose(
        harmonised_first_derivative(harmonisation_time),
        target_dspline(harmonisation_time),
    )

    # integrate to match zero-order derivative
    harmonised_spline_first_derivative_only = (
        harmonised_first_derivative.antiderivative()
    )

    # match zero-order derivatives
    diff_spline = np.subtract(
        target_spline(harmonisation_time),
        harmonised_spline_first_derivative_only(harmonisation_time),
    )

    harmonised_spline_no_convergence = add_constant_to_spline(
        in_spline=harmonised_spline_first_derivative_only, constant=diff_spline
    )

    np.testing.assert_allclose(
        harmonised_spline_no_convergence(harmonisation_time),
        target_spline(harmonisation_time),
    )

    harmonised_dspline = harmonised_spline_no_convergence.derivative()

    np.testing.assert_allclose(
        harmonised_dspline(harmonisation_time), target_dspline(harmonisation_time)
    )
