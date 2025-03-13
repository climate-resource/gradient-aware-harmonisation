from typing import Optional, Union

from gradient_aware_harmonisation.exceptions import MissingOptionalDependencyError
from gradient_aware_harmonisation.utils import Timeseries


def plotting(
    harmonisee_timeseries: Timeseries,
    target_timeseries: Timeseries,
    interpolated_timeseries: Timeseries,
    harmonisation_time: Union[int, float],
    convergence_time: Optional[Union[int, float]],
) -> None:
    """
    Plots the target, original and interpolated timeseries as computed with :func:`gradient_aware_harmonisation.harmonise.harmoniser`

    Parameters
    ----------
    harmonisee_timeseries : Timeseries
        timeseries that should be matched with the target timeseries at the harmonisation time point
    target_timeseries : Timeseries
        timeseries that is used for matching the harmonisee at the harmonisation time point
    interpolated_timeseries : Timeseries
        compute harmonised timeseries as returned by :func:`gradient_aware_harmonisation.harmonise.harmoniser`
    harmonisation_time: Union[int, float]
        time point at which the harmonisee should be matched with the target timeseries
    convergence_time: Optional[Union[int, float]]
        time point at which the harmonised timeseries should match again the original predictions of the harmonisee

    Returns
    -------
    None
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
        harmonisee_timeseries.value,
        label="harmonisee",
        linestyle="--",
        color="black",
    )
    plt.plot(
        interpolated_timeseries.time_axis,
        interpolated_timeseries.value,
        label="harmonised",
    )
    plt.plot(
        target_timeseries.time_axis,
        target_timeseries.value,
        label="target",
        color="red",
    )
    plt.axvline(harmonisation_time, color="black", linestyle="dotted")
    if convergence_time is not None:
        plt.axvline(convergence_time, color="black", linestyle="dotted")
    plt.legend(handlelength=0.3, fontsize="small", frameon=False)
    plt.show()
