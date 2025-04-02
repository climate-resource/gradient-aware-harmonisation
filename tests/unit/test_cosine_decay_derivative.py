import numpy as np
import pytest

from gradient_aware_harmonisation.convergence import (
    CosineDecaySplineHelperDerivative,
)

scipy = pytest.importorskip("scipy")


def test_CosineDecaySplineHelperDerivative():
    """
    Test correct computation of derivative
    """
    initial_time = 2002
    final_time = 2020
    apply_to_convergence = False

    cos_helper_deriv = CosineDecaySplineHelperDerivative(
        initial_time=initial_time,
        final_time=final_time,
        apply_to_convergence=apply_to_convergence,
    )

    integral, _ = scipy.integrate.quad(cos_helper_deriv, a=initial_time, b=final_time)

    np.testing.assert_allclose(integral, -1.0)
