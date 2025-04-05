"""
Harmonisation by adding a cubic
"""

from __future__ import annotations

import numpy as np

from gradient_aware_harmonisation.exceptions import MissingOptionalDependencyError
from gradient_aware_harmonisation.spline import Spline, SplineScipy, SumOfSplines


def harmonise_splines_add_cubic(
    diverge_from: Spline,
    harmonisee: Spline,
    harmonisation_time: float | int,
    convergence_time: float | int,
) -> Spline:
    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "harmonise_splines_add_cubic", requirement="scipy.interpolate"
        ) from exc

    d = diverge_from(harmonisation_time) - harmonisee(harmonisation_time)
    c = diverge_from.derivative()(harmonisation_time) - harmonisee.derivative()(
        harmonisation_time
    )

    # delta = (ct - ht)
    # 0.0 = a delta^3 + b delta^2 + c delta + d
    #
    # A.coeffs =
    # where A = ((delta^3, delta^2), (3 delta^2, 2 delta))
    # coeffs = (a b)
    # const = (-c * delta - d, -c)
    delta = convergence_time - harmonisation_time
    a_array = np.array([[delta**3, delta**2], [3 * delta**2, 2 * delta]])
    rhs = np.array([-c * delta - d, -c])

    soln = np.linalg.solve(a_array, rhs)
    a = soln[0]
    b = soln[1]

    cubic_to_add = SplineScipy(
        scipy.interpolate.PPoly(
            c=[[a, 0.0], [b, 0.0], [c, 0.0], [d, 0.0]],
            # TODO: better upper limit than 1e8
            x=[harmonisation_time, convergence_time, 1e8],
        )
    )

    res = SumOfSplines(
        harmonisee,
        cubic_to_add,
    )

    return res
