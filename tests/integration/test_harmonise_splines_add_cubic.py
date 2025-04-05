"""
Tests of harmonising by adding a cubic to the harmonisee
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_aware_harmonisation.add_cubic import harmonise_splines_add_cubic
from gradient_aware_harmonisation.spline import Spline, SplineScipy


def check_expected_continuity(
    solution: Spline,
    diverge_from: Spline,
    harmonisee: Spline,
    harmonisation_time: float,
    convergence_time: float,
) -> None:
    np.testing.assert_allclose(
        solution(harmonisation_time),
        diverge_from(harmonisation_time),
        err_msg=(
            "Difference in absolute value of solution and diverge_from "
            "at harmonisation_time"
        ),
    )
    np.testing.assert_allclose(
        solution.derivative()(harmonisation_time),
        diverge_from.derivative()(harmonisation_time),
        err_msg=(
            "Difference in gradient of solution and diverge_from "
            "at harmonisation_time"
        ),
    )

    np.testing.assert_allclose(
        solution(convergence_time),
        harmonisee(convergence_time),
        err_msg=(
            "Difference in absolute value of solution and harmonisee "
            "at convergence_time"
        ),
    )
    np.testing.assert_allclose(
        solution.derivative()(convergence_time),
        harmonisee.derivative()(convergence_time),
        err_msg=(
            "Difference in gradient of solution and harmonisee at convergence_time"
        ),
    )


def check_res_compared_to_exp(  # noqa: PLR0913
    res: Spline,
    exp: Spline,
    harmonisation_time: float,
    convergence_time: float,
    n_points: int,
    extension_factor: float = 3.0,
) -> None:
    if harmonisation_time <= convergence_time:
        time_check_base = np.linspace(
            harmonisation_time,
            convergence_time
            + (convergence_time - harmonisation_time) * extension_factor,
            n_points,
        )
    else:
        time_check_base = np.linspace(
            convergence_time
            - (harmonisation_time - convergence_time) * extension_factor,
            harmonisation_time,
            n_points,
        )

    # Include the harmonisation and convergence time in all checks
    time_check = np.union1d(
        time_check_base, np.array([harmonisation_time, convergence_time])
    )

    np.testing.assert_allclose(
        res(time_check), exp(time_check), err_msg="Difference in absolute value"
    )
    np.testing.assert_allclose(
        res.derivative()(time_check),
        exp.derivative()(time_check),
        err_msg="Difference in gradients",
    )


def test_basic_case():
    """
    A basic case that can be solved by hand

    Our timeseries to diverge from/harmonise to is:

    y_{df} = x

    Our harmonisee is

    y_{he} = 0.5x - 1

    We want to harmonise at x = 0
    and converge at x = 1.

    We are going to add

    y_a = ax^3 + bx^2 + cx + d

    to our harmonisee, to create a harmonised timeseries,
    y_{ha} = y_{he} + y_a, such that:

    1. at x = 0, y_{df} = y_{ha}
       (zero-order continuous with the timeseries to diverge from
       at the harmonisation point)
    1. at x = 0, dy_{df} / dx = dy_{ha} / dx
       (first-order continuous with the timeseries to diverge from
       at the harmonisation point)
    1. at x = 1, y_{he} = y_{ha}
       (zero-order continuous with the harmonisee
       at the convergence point)
    1. at x = 1, dy_{he} / dx = dy_{ha} / dx
       (first-order continuous with the harmonisee
       at the convergence point)

    From the first condition, we have

    y_{df}(x=0) = 0 = y_{he}(x=0) + d
    0 = -1 + d
    d = 1

    From the second condition, we have

    dy_{df}/dx(x=0) = 1 = dy_{he}/dx(x=0) + c
    1 = 0.5 + c
    c = 0.5

    From the third condition, we have

    y_{he}(x=1) = y_{he}(x=1) + y_a(x=1)
    y_a(x=1) = 0
    a + b + c + d = 0
    a + b = -0.5 - 1 = -1.5

    From the final condition, we have

    dy_{he}/dx(x=1) = dy_{he}/dx(x=1) + dy_a/dx(x=1)
    dy_a/dx(x=1) = 0
    3a + 2b + c = 0
    3a + 2b = -0.5

    From this, we can solve
    (subtract the upper equation multipled by two from the lower equation)

    a = 2.5
    b = -4

    So, we expect our harmonised spline to be:

    y = 0.5x - 1 + 2.5x^3 - 4x^2 + 0.5x + 1
    y = 2.5x^3 - 4x^2 + x
    """
    scipy = pytest.importorskip("scipy")

    harmonisation_time = 0.0
    convergence_time = 1.0

    diverge_from = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[1.0], [0.0]],
            x=[0.0, 1e8],
        )
    )

    harmonisee = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[0.5], [-1.0]],
            x=[0.0, 1e8],
        )
    )

    res = harmonise_splines_add_cubic(
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    # Given solution between x = 0 and x = 1,
    # harmonisee therafter
    exp = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[2.5, 0.0], [-4.0, 0.0], [1.0, 0.5], [0.0, -0.5]],
            x=[harmonisation_time, convergence_time, 1e8],
        )
    )

    check_expected_continuity(
        solution=res,
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    for n_points in [4, 101]:
        check_res_compared_to_exp(
            res=res,
            exp=exp,
            harmonisation_time=harmonisation_time,
            convergence_time=convergence_time,
            n_points=n_points,
        )


def test_basic_case_harmonisation_time_greater_than_convergence_time():
    """
    A basic case that can be solved by hand

    Our timeseries to diverge from/harmonise to is:

    y_{df} = x

    Our harmonisee is

    y_{he} = 0.5x - 1

    We want to harmonise at x = 1
    and converge at x = -1.

    We are going to add

    y_a = ax^3 + bx^2 + cx + d

    to our harmonisee, to create a harmonised timeseries,
    y_{ha} = y_{he} + y_a, such that:

    1. at x = 1, y_{df} = y_{ha}
       (zero-order continuous with the timeseries to diverge from
       at the harmonisation point)
    1. at x = 1, dy_{df} / dx = dy_{ha} / dx
       (first-order continuous with the timeseries to diverge from
       at the harmonisation point)
    1. at x = -1, y_{he} = y_{ha}
       (zero-order continuous with the harmonisee
       at the convergence point)
    1. at x = -1, dy_{he} / dx = dy_{ha} / dx
       (first-order continuous with the harmonisee
       at the convergence point)

    From the first condition, we have

    y_{df}(x=1) = 1 = y_{he}(x=1) + d
    1 = -0.5 + a + b + c + d
    a + b + c + d = 1.5

    From the second condition, we have

    dy_{df}/dx(x=1) = 1 = dy_{he}/dx(x=1) + 3a + 2b + c
    1 = 0.5 + 3a + 2b + c
    3a + 2b + c = 0.5

    From the third condition, we have

    y_{he}(x=-1) = y_{he}(x=-1) + y_a(x=-1)
    y_a(x=-1) = 0
    -a + b + -c + d = 0

    From the final condition, we have

    dy_{he}/dx(x=-1) = dy_{he}/dx(x=-1) + dy_a/dx(x=-1)
    dy_a/dx(x=-1) = 0
    3a - 2b + c = 0

    From this, we can solve (with e.g numpy) to get

    a = -0.25
    b = 0.125
    c = 1.0
    d = 0.625

    So, we expect our harmonised spline to be:

    y = 0.5x - 1 - 0.25x^3 + 0.125x^2 + 1.0x + 0.625
    y = - 0.25x^3 + 0.125x^2 + 1.5x - 0.375

    If we translate as needed for scipy,
    we translate into x' = x + 1
    (when x = -1, x' = 0 which is what scipy expects on the left edge)
    (there are smarter ways to do this, but this is fine and already written)
    y = -0.25(x' - 1)^3 + 0.125(x' - 1)^2 + 1.5(x' - 1) - 0.375
      = -0.25x^3 + 0.75x'^2  - 0.75x' + 0.25
                 + 0.125x'^2 - 0.25x' + 0.125
                             + 1.5x'  - 1.5
                                      - 0.375
      = -0.25 (x' + 1)^3 + 0.875(x' + 1)^2 + 0.5x' - 1.5
    """
    scipy = pytest.importorskip("scipy")

    harmonisation_time = 1.0
    convergence_time = -1.0

    # y = x
    # TODO: from left-edge or something here
    diverge_from = SplineScipy(
        scipy.interpolate.PPoly(
            # These are the constants you need given how PPoly is defined
            # (it's basically y = f(x - x_le),
            # where x_le is the left-edge of the boundary)
            c=[[1.0], [-10.0]],
            x=[-10.0, 10.0],
        )
    )

    # y = 0.5x - 1
    harmonisee = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[0.5], [-6.0]],
            x=[-10.0, 10.0],
        )
    )

    res = harmonise_splines_add_cubic(
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    # Given solution between x = -1 and x = 1,
    # harmonisee therafter (in this case before)
    exp = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[0.0, -0.25], [0.0, 0.875], [0.5, 0.5], [-6.0, -1.5]],
            x=[-10.0, convergence_time, harmonisation_time],
        )
    )

    check_expected_continuity(
        solution=res,
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    for n_points in [4, 101]:
        check_res_compared_to_exp(
            res=res,
            exp=exp,
            harmonisation_time=harmonisation_time,
            convergence_time=convergence_time,
            n_points=n_points,
        )


def test_no_change():
    """
    An edge case where the harmonisation doesn't actually need to do anything

    Our timeseries to diverge from/harmonise to is:

    y = x^2

    Our harmonisee is

    y = 2x - 1

    We want to harmonise at x = 0
    and converge at x = 1.

    In practice, this means our harmonised timeseries could just be:

    y = x^2, 0 <= x <=1
    y = 2x - 1, 1 < x

    So, we expect the algorithm to add:

    y = x^2 - 2x + 1

    to our harmonisee in other to create our
    harmonised timeseries over the harmonisation period.
    """
    scipy = pytest.importorskip("scipy")

    harmonisation_time = 0.0
    convergence_time = 1.0

    diverge_from = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[1.0], [0.0], [0.0]],
            x=[0.0, 1e8],
        )
    )

    harmonisee = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[2.0], [-1.0]],
            x=[0.0, 1e8],
        )
    )

    res = harmonise_splines_add_cubic(
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    # Given solution between x = 0 and x = 1,
    # harmonisee therafter
    exp = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[1.0, 0.0], [0.0, 2.0], [0.0, 1.0]],
            x=[0.0, 1.0, 1e8],
        )
    )

    check_expected_continuity(
        solution=res,
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    for n_points in [4, 101]:
        check_res_compared_to_exp(
            res=res,
            exp=exp,
            harmonisation_time=harmonisation_time,
            convergence_time=convergence_time,
            n_points=n_points,
        )


@pytest.mark.parametrize(
    "gradient_factor",
    (
        pytest.param(0.5, id="less_than_diverge_from"),
        pytest.param(1.0, id="same_as_diverge_from"),
        pytest.param(1.5, id="more_than_diverge_from"),
    ),
)
@pytest.mark.parametrize(
    "y_intercept_shift",
    (
        pytest.param(-0.5, id="less_than_diverge_from"),
        pytest.param(0.0, id="same_as_diverge_from"),
        pytest.param(0.25, id="more_than_diverge_from"),
    ),
)
def test_combos(gradient_factor, y_intercept_shift):
    """
    Test that the harmonisation works over a variety of cases

    These cover all possible combinations of gradient and absolute value
    of the harmonisee being higher, lower or the same
    as the timeseries to diverge from at the harmonisation time.
    """
    scipy = pytest.importorskip("scipy")

    harmonisation_time = 0.0
    convergence_time = 10.0

    # y = 2.5x + 1
    diverge_from_gradient = 2.5
    diverge_from_y_intercept = 1.0
    diverge_from = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[diverge_from_gradient], [diverge_from_y_intercept]],
            x=[0, 1e8],
        )
    )

    harmonisee = SplineScipy(
        scipy.interpolate.PPoly(
            c=[
                [diverge_from_gradient * gradient_factor],
                [diverge_from_y_intercept + y_intercept_shift],
            ],
            x=[0, 1e8],
        )
    )

    res = harmonise_splines_add_cubic(
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    check_expected_continuity(
        solution=res,
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )


@pytest.mark.parametrize(
    "harmonisation_time, convergence_time",
    (
        pytest.param(0.0, 1.0),
        pytest.param(0.0, 1.7),
        pytest.param(3.0, 8.0),
        pytest.param(-3.0, 0.0),
        pytest.param(-3.0, 8.0),
        pytest.param(-3.0, -1.0),
        pytest.param(3.0, 1.0, id="backwards_harmonisation_positive_times"),
        pytest.param(
            3.0, -1.0, id="backwards_harmonisation_positive_and_negative_time"
        ),
        pytest.param(-30.0, -10.0, id="backwards_harmonisation_negative_times"),
    ),
)
def test_harmonisation_convergence_times(harmonisation_time, convergence_time):
    """
    Test over a variety of harmonisation and convergence times
    """
    scipy = pytest.importorskip("scipy")

    diverge_from = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[2.75], [1.2]],
            x=[-100, 100],
        )
    )

    harmonisee = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[2.3], [0.5]],
            x=[-100, 100],
        )
    )

    res = harmonise_splines_add_cubic(
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    check_expected_continuity(
        solution=res,
        diverge_from=diverge_from,
        harmonisee=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )
