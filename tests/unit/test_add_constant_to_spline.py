"""
Tests of `add_constant_to_spline`
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_aware_harmonisation.spline import SplineScipy
from gradient_aware_harmonisation.utils import add_constant_to_spline

scipy = pytest.importorskip("scipy")


def test_add_constant_to_spline():
    x_rge = np.arange(2010, 2030, 1)
    y_rge = np.arange(250, 270, 1)
    constant = 2.0

    spline = SplineScipy(scipy.interpolate.make_interp_spline(x=x_rge, y=y_rge))

    res_spline = add_constant_to_spline(in_spline=spline, constant=constant)

    np.testing.assert_allclose(res_spline(x_rge), y_rge + constant)
