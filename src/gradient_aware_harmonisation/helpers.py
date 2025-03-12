import matplotlib.pyplot as plt


def plotting(
    harmonisee_timeseries,
    target_timeseries,
    interpolated_timeseries,
    harmonisation_time,
    convergence_time,
):
    plt.figure(figsize=(6, 3))
    plt.plot(
        harmonisee_timeseries.time_axis,
        harmonisee_timeseries.value,
        label="pred-orig",
        linestyle="--",
        color="black",
    )
    plt.plot(
        interpolated_timeseries.time_axis,
        interpolated_timeseries.value,
        label="pred-intpol",
    )
    plt.plot(
        target_timeseries.time_axis,
        target_timeseries.value,
        label="historical",
        color="red",
    )
    plt.axvline(harmonisation_time, color="black", linestyle="dotted")
    if convergence_time is not None:
        plt.axvline(convergence_time, color="black", linestyle="dotted")
    plt.legend(handlelength=0.3, fontsize="small", frameon=False, loc="lower left")
    plt.show()
