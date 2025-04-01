"""
Integration tests of `gradient_aware_harmonisation.convergence`
"""

import numpy as np
import pytest

from gradient_aware_harmonisation.convergence import (
    CosineDecaySplineHelper,
    get_cosine_decay_harmonised_spline,
)
from gradient_aware_harmonisation.spline import SplineScipy


def test_cosine_decay_spline():
    """
    test gamma for different time values (both single value and array)

    + gamma = 1 for gamma <= harmonisation_time
    + gamma = 0 for gamma >= convergence_time
    + gamma = cosine_decay(x) for harmonisation_time < gamma < convergence_time
    """
    spline = CosineDecaySplineHelper(
        initial_time=2.0,
        final_time=10.0,
    )

    # Before the decay
    np.testing.assert_equal(spline(1.0), 1.0)
    # After the decay
    np.testing.assert_equal(spline(11.0), 0.0)
    # In the decay
    np.testing.assert_equal(spline(6.0), 0.5)
    np.testing.assert_equal(spline(3.0), 0.5 * (1 + np.cos(np.pi * 1.0 / 8.0)))

    # Array input
    np.testing.assert_equal(
        np.array([1.0, 1.0, 0.5 * (1 + np.cos(np.pi * 1.0 / 8.0)), 0.5, 0.0, 0.0]),
        spline(np.array([1.5, 2.0, 3.0, 6.0, 10.0, 11.1])),
    )


def test_cosine_decay_spline_apply_to_convergence():
    """
    test (1-gamma) for different time values (both single value and array)

    + gamma = 1-1=0 for gamma <= harmonisation_time
    + gamma = 1-0=0 for gamma >= convergence_time
    + gamma = 1-cosine_decay(x) for harmonisation_time < gamma < convergence_time

    """
    spline = CosineDecaySplineHelper(
        initial_time=2.0, final_time=10.0, apply_to_convergence=True
    )

    # Before the decay
    np.testing.assert_equal(spline(1.0), 0.0)
    # After the decay
    np.testing.assert_equal(spline(11.0), 1.0)
    # In the decay
    np.testing.assert_equal(spline(6.0), 0.5)
    np.testing.assert_equal(
        spline(3.0), 1.0 - (0.5 * (1.0 + np.cos(np.pi * 1.0 / 8.0)))
    )

    # Array input
    np.testing.assert_equal(
        np.array(
            [0.0, 0.0, 1.0 - 0.5 * (1.0 + np.cos(np.pi * 1.0 / 8.0)), 0.5, 1.0, 1.0]
        ),
        spline(np.array([1.5, 2.0, 3.0, 6.0, 10.0, 11.1])),
    )


def test_cosine_decay_spline_sum():
    """
    test decay for gamma and (1-gamma)

    expected_res = gamma + (1-gamma) = 1

    """
    spline = CosineDecaySplineHelper(
        initial_time=2.0, final_time=10.0, apply_to_convergence=False
    )
    spline_to_convergence = CosineDecaySplineHelper(
        initial_time=2.0, final_time=10.0, apply_to_convergence=True
    )

    x_vals = np.linspace(-5.0, 15.0, 100)
    np.testing.assert_equal(1.0, spline(x_vals) + spline_to_convergence(x_vals))


def test_cosine_decay_spline_derivative():
    """
    test derivative gamma for different time values (both single value and array)

    + gamma = 0 harmonisation_time > gamma > convergence_time
    + gamma = cosine_decay_derivative(x) for harm_time < gamma < conv_time

    """
    spline = CosineDecaySplineHelper(
        initial_time=2.0,
        final_time=10.0,
    ).derivative()

    # Before the decay
    np.testing.assert_equal(spline(1.0), 0.0)
    # After the decay
    np.testing.assert_equal(spline(11.0), 0.0)
    # In the decay
    np.testing.assert_equal(spline(6.0), -0.5)
    np.testing.assert_equal(spline(3.0), -0.5 * np.sin(np.pi * 1.0 / 8.0))

    # Array input
    np.testing.assert_equal(
        np.array([0.0, 0.0, -0.5 * np.sin(np.pi * 1.0 / 8.0), -0.5, 0.0, 0.0]),
        spline(np.array([1.5, 2.0, 3.0, 6.0, 10.0, 11.1])),
    )


def test_cosine_decay_spline_derivative_apply_to_convergence():
    """
    test derivative (1-gamma) for different time values (both single value and array)

    + gamma = 0 harmonisation_time > gamma > convergence_time
    + gamma = -cosine_decay_derivative(x) for harm_time < gamma < conv_time

    """
    spline = CosineDecaySplineHelper(
        initial_time=2.0, final_time=10.0, apply_to_convergence=True
    ).derivative()

    # Before the decay
    np.testing.assert_equal(spline(1.0), 0.0)
    # After the decay
    np.testing.assert_equal(spline(11.0), 0.0)
    # In the decay
    np.testing.assert_equal(spline(6.0), 0.5)
    np.testing.assert_equal(spline(3.0), 0.5 * np.sin(np.pi * 1.0 / 8.0))

    # Array input
    np.testing.assert_equal(
        np.array([0.0, 0.0, 0.5 * np.sin(np.pi * 1.0 / 8.0), 0.5, 0.0, 0.0]),
        spline(np.array([1.5, 2.0, 3.0, 6.0, 10.0, 11.1])),
    )


def test_cosine_decay_spline_derivative_sum():
    """
    test derivative decay for gamma and (1-gamma)

    expected_res = gamma + (-gamma) = 0

    """
    spline = CosineDecaySplineHelper(
        initial_time=2.0, final_time=10.0, apply_to_convergence=False
    ).derivative()

    spline_to_convegence = CosineDecaySplineHelper(
        initial_time=2.0, final_time=10.0, apply_to_convergence=True
    ).derivative()

    x_vals = np.linspace(-5.0, 15.0, 100)
    np.testing.assert_equal(0.0, spline(x_vals) + spline_to_convegence(x_vals))


def test_get_cosine_decay_harmonised_spline():
    """
    test weighted sum of two splines

    gamma * spline1  + (1-gamma) * spline2 with
    numerator = (pi * x - harmonisation_time)
    denominator = (convergence_time - harmonisation_time)
    gamma = 0.5 * (1 + cos(numerator/denominator)

    """
    scipy = pytest.importorskip("scipy")

    x_min = 0.0
    x_max = 10.0

    harmonisation_time = 2.5
    convergence_time = 8.5

    x_up_to_harmonisation_time = np.linspace(x_min, harmonisation_time, 50)
    x_after_convergence_time = np.linspace(convergence_time, x_max, 50)

    harmonised_spline_no_convergence = SplineScipy(
        scipy.interpolate.PPoly(
            x=[x_min, x_max],
            c=[[1], [0], [0], [0]],  # y=x^3
        )
    )
    convergence_spline = SplineScipy(
        scipy.interpolate.PPoly(
            x=[x_min, x_max],
            c=[[-1], [1], [2]],  # y=-x^2 + x + 2
        )
    )

    res = get_cosine_decay_harmonised_spline(
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
        harmonised_spline_no_convergence=harmonised_spline_no_convergence,
        convergence_spline=convergence_spline,
    )

    np.testing.assert_equal(
        harmonised_spline_no_convergence(x_up_to_harmonisation_time),
        res(x_up_to_harmonisation_time),
    )

    np.testing.assert_equal(
        convergence_spline(x_after_convergence_time),
        res(x_after_convergence_time),
    )

    # Times in between harmonisation and convergence times
    np.testing.assert_equal(
        np.array(
            [
                0.5
                * (1.0 + np.cos(np.pi * 0.5 / 6.0))
                * harmonised_spline_no_convergence(3.0)
                + (1.0 - 0.5 * (1.0 + np.cos(np.pi * 0.5 / 6.0)))
                * convergence_spline(3.0),
                0.5 * harmonised_spline_no_convergence(5.5)
                + 0.5 * convergence_spline(5.5),
                0.5
                * (1.0 + np.cos(np.pi * 3.5 / 6.0))
                * harmonised_spline_no_convergence(6.0)
                + (1.0 - 0.5 * (1.0 + np.cos(np.pi * 3.5 / 6.0)))
                * convergence_spline(6.0),
            ]
        ),
        res(np.array([3.0, 5.5, 6.0])),
    )
