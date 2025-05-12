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
    """
    Compute Taylor shift according to Shaw and Traub (1974)

    Parameters
    ----------
    coeffs
        polynomial coefficients

    shift
        Taylor shift

    Returns
    -------
    :
        new polynomial coefficients by Taylor shift

    References
    ----------
    Shaw, M., & Traub, J. F. (1974). On the number of multiplications
    for the evaluation of a polynomial and some of its derivatives.
    Journal of the ACM, 21(1), 161-167.
    """
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
    """
    Generate cubic spline

    The cubic spline interpolates between harmonised
    spline and harmonisee

    Parameters
    ----------
    diverge_from
        Spline whose value and first derivative at the
        harmonization time match the target spline values.

    harmonisee
        Spline that the cubic spline should converge to
        and match at and after the convergence time.

    harmonisation_time
        Time point at which cubic spline should be
        matched to the target.

    convergence_time
        Time point at which the cubic spline should
        converge to harmonisee.

    Returns
    -------
    :
        cubic spline

    Examples
    --------
    >>> from gradient_aware_harmonisation.timeseries import Timeseries
    >>> from gradient_aware_harmonisation.add_cubic import harmonise_splines_add_cubic
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt

    >>> harmonisation_time = 3.2
    >>> convergence_time = 3.8
    >>> time = np.arange(3.0, 4.1, step=0.1)

    >>> harmonised = Timeseries(time_axis=time, values=-0.5 * time + 1)
    >>> harmonisee = Timeseries(time_axis=time, values=0.2 * time**3 - 5)
    >>> cubic_spline = harmonise_splines_add_cubic(
    ...     diverge_from=harmonised.to_spline(),
    ...     harmonisee=harmonisee.to_spline(),
    ...     harmonisation_time=harmonisation_time,
    ...     convergence_time=convergence_time,
    ... )

    >>> _, ax = plt.subplots(figsize=(6, 3))
    >>> for y, name in zip(
    ...     [harmonised.values, harmonisee.values, cubic_spline(time)],
    ...     ["diverge_from", "harmonisee", "cubic_spline"],
    ... ):
    ...     ax.plot(time, y, label=name)  # doctest: +SKIP
    >>> ax.axvline(harmonisation_time, color="black", linestyle="--")  # doctest: +SKIP
    >>> ax.axvline(convergence_time, color="black", linestyle="--")  # doctest: +SKIP
    >>> ax.legend()  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    ![](../../../assets/images/cubic_spline.png)

    """
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
