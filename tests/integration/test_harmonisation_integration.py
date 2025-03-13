"""
Integration tests of the `harmonise` module
"""

import numpy as np
import pytest

from gradient_aware_harmonisation.harmonise import harmoniser
from gradient_aware_harmonisation.utils import Timeseries


@pytest.mark.parametrize("harmonisation_time", [2015, 2016, 2017])
@pytest.mark.parametrize("convergence_time", [None, 2030, 2050])
def test_already_harmonised_remains_unchanged(harmonisation_time, convergence_time):
    target = Timeseries(time_axis=np.arange(2010, 2017), values=np.arange(7))
    harmonisee = Timeseries(
        time_axis=np.arange(2015, 2100),
        values=np.hstack(([4, 5, 6], np.arange(2100 - 2017))),
    )

    assert harmonisation_time in target.time_axis, "Your test will not work"
    assert harmonisation_time in harmonisee.time_axis, "Your test will not work"
    assert (
        convergence_time is None or convergence_time in harmonisee.time_axis
    ), "Your test will not work"

    res = harmoniser(
        target_timeseries=target,
        harmonisee_timeseries=harmonisee,
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
    )

    # We expect to get out what we put in as it's already harmonised
    exp = harmonisee

    np.testing.assert_allclose(res.time_axis, exp.time_axis)
    np.testing.assert_allclose(res.value, exp.value)
