"""
Integration tests of `harmonise_splines` function in `utils` module

In this module, we need to test a few things:

- different configurations
    - are the target and the harmonisee already harmonised or not?
    - do we specify a convergence time or not?
    - do we do harmonisation at the boundary of our timeseries
      (or even request harmonisation on a timepoint not in our timeseries)?
    - more complicated/realistic time axes (e.g. not integer steps)

- both zeroth-order and first-order continuity in all cases
"""

import numpy as np
import pytest

from gradient_aware_harmonisation.utils import Splines, Timeseries, harmonise_splines

scipy = pytest.importorskip("scipy")

test_criteria = pytest.mark.parametrize("test_criterion", ("zero-order", "first-order"))


def check_continuity(  # noqa: PLR0913
    test_criterion, harmonisation_time, harmonised, target, rtol=1e-8, atol=0.0
):
    if test_criterion == "zero-order":
        # test absolute value
        np.testing.assert_allclose(
            harmonised(harmonisation_time),
            target(harmonisation_time),
            rtol=rtol,
            atol=atol,
            err_msg="Difference in zero-order values",
        )
        # assert pytest.approx(harmon_spline(harmonisation_time)) == splines.target(
        #     harmonisation_time
        # ), (
        #     "target and harmonised spline have not the "
        #     + f"same absolute value at {harmonisation_time=}. "
        #     + f"Got harmonisee: {harmon_spline(harmonisation_time)}"
        #     + f" vs. target: {splines.target(harmonisation_time)}."
        # )

    elif test_criterion == "first-order":
        # test first derivative
        np.testing.assert_allclose(
            harmonised.derivative()(harmonisation_time),
            target.derivative()(harmonisation_time),
            rtol=rtol,
            atol=atol,
            err_msg="Difference in first-derivative",
        )
        # # compute first-order derivative
        #
        # # prepare error message
        # msg = (
        #     "target and harmonised spline have not the "
        #     + f"same first-order derivative at the {harmonisation_time=}.\n"
        #     + f"Got harmonisee: {d_harmon_spline(harmonisation_time)} vs. "
        #     + f"target: {d_target_spline(harmonisation_time)}"
        # )
        #
        # # test first-order derivative
        # assert pytest.approx(d_harmon_spline(harmonisation_time)) == d_target_spline(
        #     harmonisation_time
        # ), msg

    else:
        raise NotImplementedError(test_criterion)


@pytest.mark.parametrize("harmonisation_time", (1.0, 3.0))
@pytest.mark.parametrize("convergence_time", (None, 3.0))
@test_criteria
def test_target_and_harmonisee_equal(
    test_criterion, convergence_time, harmonisation_time
):
    time_axis = np.array([0.0, 1.0, 2.0, 3.0])
    timeseries_target = Timeseries(time_axis=time_axis, values=time_axis**2)
    harmonisee = timeseries_target

    splines = Splines(
        target=scipy.interpolate.make_interp_spline(
            timeseries_target.time_axis, timeseries_target.values
        ),
        harmonisee=scipy.interpolate.make_interp_spline(
            harmonisee.time_axis, harmonisee.values
        ),
    )

    harmon_spline = harmonise_splines(
        splines,
        harmonisee_timeseries=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    check_continuity(
        test_criterion=test_criterion,
        harmonisation_time=harmonisation_time,
        harmonised=harmon_spline,
        target=splines.target,
    )


@pytest.mark.parametrize("harmonisation_time", (1.0, 3.0))
@pytest.mark.parametrize("convergence_time", (None, 3.0))
@test_criteria
def test_target_and_harmonisee_differ(
    test_criterion, convergence_time, harmonisation_time
):
    time_axis_target = np.array([0.0, 1.0, 2.0, 3.0])
    timeseries_target = Timeseries(
        time_axis=time_axis_target, values=time_axis_target ** (1 / 2)
    )

    time_axis_harmonisee = np.array([3.0, 4.0, 5.0, 6.0])
    timeseries_harmonisee = Timeseries(
        time_axis=time_axis_harmonisee, values=-1.3 * np.sin(time_axis_harmonisee) + 8
    )

    splines = Splines(
        target=scipy.interpolate.make_interp_spline(
            timeseries_target.time_axis, timeseries_target.values
        ),
        harmonisee=scipy.interpolate.make_interp_spline(
            timeseries_harmonisee.time_axis, timeseries_harmonisee.values
        ),
    )

    harmon_spline = harmonise_splines(
        splines,
        harmonisee_timeseries=timeseries_harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    check_continuity(
        test_criterion=test_criterion,
        harmonisation_time=harmonisation_time,
        harmonised=harmon_spline,
        target=splines.target,
    )


@pytest.mark.parametrize("harmonisation_time", (2003.0,))
@pytest.mark.parametrize("convergence_time", (None, 2005.0))
@test_criteria
def test_more_realistic(test_criterion, convergence_time, harmonisation_time):
    """
    Both testing more realistic data but also a time axis that has integer values
    """
    timeseries_target = Timeseries(
        time_axis=np.array([2000, 2001, 2002, 2003]),
        values=np.array([371.77, 373.72, 376.33, 378.43]),
    )

    timeseries_harmonisee = Timeseries(
        time_axis=np.array([2003, 2004, 2005, 2006]),
        values=np.array([376.28, 378.83, 381.20, 382.55]),
    )

    splines = Splines(
        target=scipy.interpolate.make_interp_spline(
            timeseries_target.time_axis, timeseries_target.values
        ),
        harmonisee=scipy.interpolate.make_interp_spline(
            timeseries_harmonisee.time_axis, timeseries_harmonisee.values
        ),
    )

    harmon_spline = harmonise_splines(
        splines,
        harmonisee_timeseries=timeseries_harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    check_continuity(
        test_criterion=test_criterion,
        harmonisation_time=harmonisation_time,
        harmonised=harmon_spline,
        target=splines.target,
    )


# target == harmonisee; continuous time_axis
def f_equal():
    res = dict(x=np.array([0.0, 1.0, 2.0, 3.0]))
    res["y"] = res["x"] ** 2
    return res


# target != harmonisee; continuous time_axis
def f1():
    res = dict(x=np.array([0.0, 1.0, 2.0, 3.0]))
    res["y"] = res["x"] ** (1 / 2)
    return res


def f2():
    res = dict(x=np.array([3.0, 4.0, 5.0, 6]))
    res["y"] = -1.3 * np.sin(res["x"]) + 8
    return res


# target != harmonisee; integer time_axis
def t1():
    res = dict(
        x=np.array([2000, 2001, 2002, 2003]),
        y=np.array([371.77, 373.72, 376.33, 378.43]),
    )
    return res


def t2():
    res = dict(
        x=np.array([2003, 2004, 2005, 2006]),
        y=np.array([376.28, 378.83, 381.20, 382.55]),
    )
    return res


target_equals_harmonisee = [
    pytest.param(
        harmonisation_time,
        convergence,
        f_equal,
        f_equal,
        id=f"target-equals-harmonisee-{id_conv}-{id_ht}",
    )
    for id_conv, convergence in (("convergence", 3.0), ("no-convergence", None))
    for id_ht, harmonisation_time in (("not-at-boundary", 1.0), ("at-boundary", 3.0))
]

target_differs_from_harmonisee = [
    pytest.param(
        harmonisation_time,
        convergence,
        f1,
        f2,
        id=f"target-differs-from-harmonisee-{id_conv}-{id_ht}",
    )
    for id_conv, convergence in (("convergence", 3.0), ("no-convergence", None))
    for id_ht, harmonisation_time in (("not-at-boundary", 1.0), ("at-boundary", 3.0))
]

realistic_dataset = [
    pytest.param(2003, convergence, t1, t2, id=f"integer-harmonisation-year-{id}")
    for id, convergence in (("convergence", 2005), ("no-convergence", None))
]


@pytest.mark.parametrize("test_criterion", ["zero-order", "first-order"])
@pytest.mark.parametrize(
    "harmonisation_time, convergence_time, target_func, harmonisee_func",
    [
        *target_equals_harmonisee,
        *target_differs_from_harmonisee,
        *realistic_dataset,
    ],
)
def test_harmonise_splines_equal_at_harmonisation_time(
    harmonisation_time,
    convergence_time,
    target_func,
    harmonisee_func,
    test_criterion,
):
    """
    We check both the zeroth-order and first-order continuity
    """
    scipy = pytest.importorskip("scipy")

    target = target_func()
    harmonisee = harmonisee_func()

    timeseries_target = Timeseries(time_axis=target["x"], values=target["y"])
    timeseries_harmonisee = Timeseries(
        time_axis=harmonisee["x"], values=harmonisee["y"]
    )

    splines = Splines(
        target=scipy.interpolate.make_interp_spline(
            timeseries_target.time_axis, timeseries_target.values
        ),
        harmonisee=scipy.interpolate.make_interp_spline(
            timeseries_harmonisee.time_axis, timeseries_harmonisee.values
        ),
    )

    harmon_spline = harmonise_splines(
        splines,
        harmonisee_timeseries=timeseries_harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    if test_criterion == "zero-order":
        # test absolute value
        assert pytest.approx(harmon_spline(harmonisation_time)) == splines.target(
            harmonisation_time
        ), (
            "target and harmonised spline have not the "
            + f"same absolute value at {harmonisation_time=}. "
            + f"Got harmonisee: {harmon_spline(harmonisation_time)}"
            + f" vs. target: {splines.target(harmonisation_time)}."
        )

    if test_criterion == "first-order":
        # compute first-order derivative
        d_harmon_spline = harmon_spline.derivative()
        d_target_spline = splines.target.derivative()

        # prepare error message
        msg = (
            "target and harmonised spline have not the "
            + f"same first-order derivative at the {harmonisation_time=}.\n"
            + f"Got harmonisee: {d_harmon_spline(harmonisation_time)} vs. "
            + f"target: {d_target_spline(harmonisation_time)}"
        )

        # test first-order derivative
        assert pytest.approx(d_harmon_spline(harmonisation_time)) == d_target_spline(
            harmonisation_time
        ), msg
