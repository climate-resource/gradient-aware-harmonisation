"""
Harmonisation by adding a cubic
"""

from __future__ import annotations

import numpy as np

from gradient_aware_harmonisation.exceptions import MissingOptionalDependencyError
from gradient_aware_harmonisation.spline import Spline, SplineScipy, SumOfSplines
from gradient_aware_harmonisation.typing import NP_ARRAY_OF_FLOAT_OR_INT


def taylor_shift(
    coeffs: NP_ARRAY_OF_FLOAT_OR_INT, shift: float
) -> NP_ARRAY_OF_FLOAT_OR_INT:
    if shift == 0.0:
        return coeffs

    n = coeffs.size - 1
    # Algorithm expects coefficient of x**n to be in coeffs[n], hence
    coeffs_r = coeffs[::-1]

    # There are less compute and memory intense ways to do this,
    # but we're not that worried about that right now.
    store = np.zeros((n + 1, n + 1))
    for i in range(n):
        store[i, 0] = coeffs_r[n - i - 1] * shift ** (n - i - 1)
        store[i, i + 1] = coeffs_r[n] * shift**n

    for j in range(n):
        for i in range(j + 1, n + 1):
            store[i, j + 1] = store[i - 1, j] + store[i - 1, j + 1]

    res_r = np.zeros(coeffs.size)
    for i in range(n):
        res_r[i] = store[n, i + 1] / shift**i

    res_r[n] = coeffs_r[n]

    # Return back to expected convention
    res = res_r[::-1]
    return res


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

    if convergence_time > harmonisation_time:
        delta = convergence_time - harmonisation_time
        a_matrix = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [delta**3, delta**2, delta, 1.0],
                [3 * delta**2, 2 * delta, 1, 0.0],
            ]
        )

    else:
        delta = harmonisation_time - convergence_time
        a_matrix = np.array(
            [
                [delta**3, delta**2, delta, 1.0],
                [3 * delta**2, 2 * delta, 1, 0.0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )

    rhs = np.array(
        [
            diverge_from(harmonisation_time) - harmonisee(harmonisation_time),
            diverge_from.derivative()(harmonisation_time)
            - harmonisee.derivative()(harmonisation_time),
            0.0,
            0.0,
        ]
    )

    coeffs = np.linalg.solve(a_matrix, rhs)

    if harmonisation_time <= convergence_time:
        cubic_to_add = SplineScipy(
            scipy.interpolate.PPoly(
                c=np.vstack([coeffs, np.zeros_like(coeffs)]).T,
                # TODO: better upper limit by using harmonisee.domain[1]
                x=[harmonisation_time, convergence_time, 1e8],
            )
        )

    else:
        cubic_to_add = SplineScipy(
            scipy.interpolate.PPoly(
                c=np.vstack([np.zeros_like(coeffs), coeffs]).T,
                # TODO: better lower limit by using harmonisee.domain[0]
                x=[-1e8, convergence_time, harmonisation_time],
            )
        )

    res = SumOfSplines(harmonisee, cubic_to_add)

    return res
