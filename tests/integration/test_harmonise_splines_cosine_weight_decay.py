import numpy as np
import pytest

from gradient_aware_harmonisation.convergence import get_cosine_decay_harmonised_spline
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

    res = get_cosine_decay_harmonised_spline(
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
