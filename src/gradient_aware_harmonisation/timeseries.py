"""
Definition of our timeseries class
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy.typing as npt
from attrs import define, field

from gradient_aware_harmonisation.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    # In TYPE_CHECKING guard to avoid circular imports
    from gradient_aware_harmonisation.spline import SplineScipy


@define
class Timeseries:
    """
    Timeseries class
    """

    time_axis: npt.NDArray[Any]
    values: npt.NDArray[Any] = field()

    @values.validator
    def values_validator(self, attribute: Any, value: Any) -> None:
        """
        Validate the values

        Parameters
        ----------
        attribute
            Attribute to validate

        value
            Value to validate
        """
        if value.size != self.time_axis.size:
            msg = (
                f"{attribute.name} must have the same size as time_axis. "
                f"Received {value.size=} {self.time_axis.size=}"
            )
            raise ValueError(msg)

    def to_spline(self, **kwargs: Any) -> SplineScipy:
        """
        Convert to a continuous spline

        Uses [scipy.interpolate.make_interp_spline][]
        with `self`'s values and time axis.

        Parameters
        ----------
        **kwargs
            Passed to [scipy.interpolate.make_interp_spline][]

        Returns
        -------
        :
            Spline, generated using [scipy.interpolate.make_interp_spline][]
        """
        try:
            import scipy.interpolate
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "to_spline", requirement="scipy.interpolate"
            ) from exc

        # Late import to avoid circularity
        from gradient_aware_harmonisation.spline import SplineScipy

        return SplineScipy(
            scipy.interpolate.make_interp_spline(
                x=self.time_axis, y=self.values, **kwargs
            ),
        )
