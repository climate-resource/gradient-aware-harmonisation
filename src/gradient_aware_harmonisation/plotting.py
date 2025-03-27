"""
Helper functions
"""

from typing import Optional, Union

from gradient_aware_harmonisation.exceptions import MissingOptionalDependencyError
from gradient_aware_harmonisation.timeseries import Timeseries


def plotting(
    harmonisee_timeseries: Timeseries,
    target_timeseries: Timeseries,
    interpolated_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_time: Optional[Union[int, float]],
) -> None:
    """
    Plot the target, original and interpolated timeseries

    We expect these to have been computed with
    [`harmonise`][gradient_aware_harmonisation.harmonise].

    Parameters
    ----------
    harmonisee_timeseries
        Harmonisee timeseries (i.e. the timeseries we want to harmonise)

    target_timeseries
        Target timeseries (i.e. what we harmonise to)

    interpolated_timeseries
        Harmonised timeseries as returned by
        [`harmonise`][gradient_aware_harmonisation.harmonise]

    harmonisation_time
        Time point at which harmonisee should be matched to the target

    convergence_time
        Time point at which the harmonised data
        should converge towards the prediced data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    plt.figure(figsize=(6, 3))
    plt.plot(
        harmonisee_timeseries.time_axis,
        harmonisee_timeseries.values,
        label="harmonisee",
        linestyle="--",
        color="black",
    )
    plt.plot(
        interpolated_timeseries.time_axis,
        interpolated_timeseries.values,
        label="harmonised",
    )
    plt.plot(
        target_timeseries.time_axis,
        target_timeseries.values,
        label="target",
        color="red",
    )
    plt.axvline(harmonisation_time, color="black", linestyle="dotted")
    if convergence_time is not None:
        plt.axvline(convergence_time, color="black", linestyle="dotted")
    plt.legend(handlelength=0.3, fontsize="small", frameon=False)
    plt.xlabel("time axis")
    plt.ylabel("value")
    plt.show()
