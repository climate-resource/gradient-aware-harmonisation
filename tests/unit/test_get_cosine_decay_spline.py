"""
Tests of `get_cosine_decay_harmonised_spline`
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_aware_harmonisation import get_cosine_decay_harmonised_spline
from gradient_aware_harmonisation.timeseries import Timeseries
from gradient_aware_harmonisation.utils import add_constant_to_spline

scipy = pytest.importorskip("scipy")


@pytest.mark.parametrize("harmonisation_time", [3.0, 3.5, 4.0])
@pytest.mark.parametrize("convergence_time", [6.0, 9.0, 10.0])
def test_get_cosine_decay(harmonisation_time, convergence_time):
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

    # integrate to match zero-order derivative
    harmonised_spline_first_derivative_only = (
        harmonised_first_derivative.antiderivative()
    )

    # match zero-order derivatives
    diff_spline = np.subtract(
        target_spline(harmonisation_time),
        harmonised_spline_first_derivative_only(harmonisation_time),
    )

    diverge_from = add_constant_to_spline(
        in_spline=harmonised_spline_first_derivative_only, constant=diff_spline
    )

    harmonised_spline_convergence = get_cosine_decay_harmonised_spline(
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
        diverge_from=diverge_from,
        convergence_spline=harmonisee_spline,
    )

    np.testing.assert_allclose(
        harmonised_spline_convergence(harmonisation_time),
        target_spline(harmonisation_time),
    )

    np.testing.assert_allclose(
        harmonised_spline_convergence(convergence_time),
        harmonisee_spline(convergence_time),
    )
