"""
Utility functions
"""

from __future__ import annotations

from typing import (
    Protocol,
    Union,
)

import numpy as np

from gradient_aware_harmonisation.convergence import get_cosine_decay_harmonised_spline
from gradient_aware_harmonisation.exceptions import MissingOptionalDependencyError
from gradient_aware_harmonisation.spline import (
    Spline,
    SplineScipy,
    SumOfSplines,
)
from gradient_aware_harmonisation.typing import NP_FLOAT_OR_INT, PINT_SCALAR


class GetHarmonisedSplineLike(Protocol):
    """
    A callable which can generate a final, harmonised spline

    The harmonised spline is generated based on a
    harmonised spline that doesn't consider convergence
    and a spline to which the final, harmonised spline should converge.
    """

    def __call__(
        self,
        harmonisation_time: Union[int, float],
        convergence_time: Union[int, float],
        harmonised_spline_no_convergence: Spline,
        convergence_spline: Spline,
    ) -> Spline:
        """
        Generate the harmonised spline

        Parameters
        ----------
        harmonisation_time
            Harmonisation time

            This is the time at and before which
            the solution should be equal to `harmonised_spline_no_convergence`.

        convergence_time
            Convergence time

            This is the time at and after which
            the solution should be equal to `convergence_spline`.

        harmonised_spline_no_convergence
            Harmonised spline that does not consider convergence

        convergence_spline
            The spline to which the result should converge

        Returns
        -------
        :
            Harmonised spline
        """


# this function is not used in the harmonise() function
# However, it is a useful wrapper for getting the harmonised spline
# without applying a further convergence method
def harmonise_splines(  # noqa: PLR0913
    harmonisee: Spline,
    target: Spline,
    harmonisation_time: Union[int, float],
    converge_to: Spline,
    convergence_time: Union[int, float],
    get_harmonised_spline: GetHarmonisedSplineLike = get_cosine_decay_harmonised_spline,
) -> Spline:
    """
    Harmonise spline

    Parameters
    ----------
    harmonisee
        Spline that we want to harmonise

    target
        Spline to which we harmonise

    harmonisation_time
        Time point at which harmonisee should be matched to the target

    converge_to
        The spline to which the result should converge.

        If not supplied, we use `harmonisee'.
        i.e. we converge back to the spline we are harmonising.

    convergence_time
        Time point at which the harmonised data
        should converge to the convergence spline

    get_harmonised_spline
        The method to use to converge back to the convergence spline.

    Returns
    -------
    :
        harmonised spline
    """
    # compute derivatives of splines
    target_dspline = target.derivative()
    harmonisee_dspline = harmonisee.derivative()

    # match first-order derivatives
    diff_dspline = np.subtract(
        target_dspline(harmonisation_time), harmonisee_dspline(harmonisation_time)
    )

    harmonised_first_derivative = add_constant_to_spline(
        in_spline=harmonisee_dspline, constant=diff_dspline
    )

    # integrate to match zero-order derivative
    harmonised_spline_first_derivative_only = (
        harmonised_first_derivative.antiderivative()
    )

    # match zero-order derivatives
    diff_spline = np.subtract(
        target(harmonisation_time),
        harmonised_spline_first_derivative_only(harmonisation_time),
    )

    harmonised_spline_no_convergence = add_constant_to_spline(
        in_spline=harmonised_spline_first_derivative_only, constant=diff_spline
    )

    harmonised_spline = get_harmonised_spline(
        harmonisation_time=harmonisation_time,
        convergence_time=convergence_time,
        harmonised_spline_no_convergence=harmonised_spline_no_convergence,
        convergence_spline=converge_to,
    )

    return harmonised_spline


def add_constant_to_spline(in_spline: Spline, constant: float | int) -> Spline:
    """
    Add a constant value to a spline

    Parameters
    ----------
    in_spline
        Input spline

    constant
        Constant to add

    Returns
    -------
    :
        Spline plus the given constant
    """
    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "add_constant_to_spline", requirement="scipy"
        ) from exc

    return SumOfSplines(
        spline_one=in_spline,
        spline_two=SplineScipy(
            scipy.interpolate.PPoly(
                c=[[constant]],
                # # TODO: Problem: Currently domain is defined for SumOfSplines
                # #  and SplineScipy should be specified only once
                # #  preferably in SplineScipy
                # x=in_spline.domain,
                # TODO: better solution for domain handling
                x=[-1e8, 1e8],
            )
        ),
    )


def validate_domain(
    domain: Union[
        tuple[PINT_SCALAR, PINT_SCALAR], tuple[NP_FLOAT_OR_INT, NP_FLOAT_OR_INT]
    ],
) -> None:
    """
    Check that domain values are valid

    Parameters
    ----------
    domain
        Domain to check

    Raises
    ------
    AssertionError
        `len(domain) != 2` or `domain[1] <= domain[0]`.
    """
    expected_domain_length = 2
    if len(domain) != expected_domain_length:
        raise AssertionError(len(domain))

    if domain[1] <= domain[0]:
        msg = f"domain[1] must be greater than domain[0]. Received {domain=}."

        raise AssertionError(msg)
