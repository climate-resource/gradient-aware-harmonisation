import numpy as np
import pytest

from gradient_aware_harmonisation.convergence import (
    LogisticDecaySplineHelperDerivative,
)

scipy = pytest.importorskip("scipy")


def test_LogisticDecaySplineHelperDerivative():
    """
    Test correct computation of derivative
    """
    initial_time = 2002
    final_time = 2020
    apply_to_convergence = False

    logistic_helper_deriv = LogisticDecaySplineHelperDerivative(
        initial_time=initial_time,
        final_time=final_time,
        apply_to_convergence=apply_to_convergence,
        slope=np.exp(0.0),
        shift=0.0,
    )

    integral, _ = scipy.integrate.quad(
        logistic_helper_deriv, a=initial_time, b=final_time
    )

    np.testing.assert_allclose(integral, -1.0)
