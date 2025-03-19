"""
Integration tests of the `harmonise` module
"""

import numpy as np
import pytest

from gradient_aware_harmonisation import harmonise
from gradient_aware_harmonisation.utils import Timeseries


# TODO: add harmonisation_time halfway between points in time axis
@pytest.mark.parametrize("harmonisation_time", [2015, 2016, 2017])
@pytest.mark.parametrize("convergence_time", [None, 2030, 2050])
def test_already_harmonised_remains_unchanged(harmonisation_time, convergence_time):
    pytest.importorskip("scipy")
    # Note: you have to be very careful here to make sure
    # that the target and harmonisee are the same in both absolute value and gradient,
    # even once converted to a spline.
    # (Hence the very simple set up below, it is tricky to ensure
    # this equality otherwise)
    target = Timeseries(time_axis=np.arange(2015, 2100), values=np.arange(2100 - 2015))
    harmonisee = target

    assert harmonisation_time in target.time_axis, "Your test will not work"
    assert harmonisation_time in harmonisee.time_axis, "Your test will not work"
    assert (
        convergence_time is None or convergence_time in harmonisee.time_axis
    ), "Your test will not work"

    np.testing.assert_equal(
        target.values[target.time_axis == harmonisation_time],
        harmonisee.values[harmonisee.time_axis == harmonisation_time],
    )

    res = harmonise(
        target_timeseries=target,
        harmonisee_timeseries=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    # We expect to get out what we put in as it's already harmonised
    exp_indexer = np.where(harmonisee.time_axis >= harmonisation_time)
    exp = Timeseries(
        time_axis=harmonisee.time_axis[exp_indexer],
        values=harmonisee.values[exp_indexer],
    )

    np.testing.assert_allclose(res.time_axis, exp.time_axis)
    np.testing.assert_allclose(res.values, exp.values)
