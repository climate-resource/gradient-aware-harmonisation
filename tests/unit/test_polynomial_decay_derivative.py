import numpy as np
import pytest

from gradient_aware_harmonisation.convergence import (
    PolynomialDecaySplineHelperDerivative,
)

scipy = pytest.importorskip("scipy")


def test_PolynomialDecaySplineHelperDerivative():
    """
    Test correct computation of derivative
    """
    initial_time = 2002
    final_time = 2020
    apply_to_convergence = False

    cos_helper_deriv = PolynomialDecaySplineHelperDerivative(
        initial_time=initial_time,
        final_time=final_time,
        apply_to_convergence=apply_to_convergence,
        power=2.0,
    )

    integral, _ = scipy.integrate.quad(cos_helper_deriv, a=initial_time, b=final_time)

    np.testing.assert_allclose(integral, -1.0)
