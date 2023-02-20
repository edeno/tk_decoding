import numpy as np
import pandas as pd
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)
from scipy.ndimage import gaussian_filter1d
from src.parameters import SAMPLING_FREQUENCY, CM_PER_PIXEL


def load_data():
    position_info = get_position_info()
    spike_times = get_spike_times()

    spikes = convert_spike_times_to_indicator(spike_times, position_info.index)

    is_good_position = np.all(~np.isnan(position_info), axis=1)
    position_info = position_info.loc[is_good_position]
    spikes = spikes.loc[is_good_position]

    multiunit_firing_rate = get_multiunit_population_firing_rate(
        multiunit=spikes.to_numpy(),
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    multiunit_firing_rate = pd.DataFrame(
        multiunit_firing_rate, index=position_info.index
    )

    multiunit_HSE_times = multiunit_HSE_detector(
        time=position_info.index.to_numpy(),
        multiunit=spikes.to_numpy(),
        speed=position_info.speed.to_numpy(),
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    return position_info, spikes, multiunit_firing_rate, multiunit_HSE_times


def get_position_info():
    position_info = (
        pd.read_csv("../Raw-Data/position4Xulu.csv")[["x", "y"]] * CM_PER_PIXEL
    )

    time = np.arange(len(position_info)) / SAMPLING_FREQUENCY
    position_info = position_info.set_index(pd.Index(time, name="time"))

    position_info["speed"] = get_speed(
        position=position_info.to_numpy(),
        time=position_info.index.to_numpy(),
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    return position_info


def get_spike_times():
    return (
        pd.read_csv("../Raw-Data/df4Xulu.csv")
        .loc[:, ["channel", "cluster_ID", "photometry_timestamp_250Hz"]]
        .astype(int)
        .set_index(["channel", "cluster_ID"])
        .sort_index()
    )


def convert_spike_times_to_indicator(spike_times, time):
    n_time = len(time)
    n_cells = len(spike_times.index.unique())

    spikes = np.zeros((n_time, n_cells), dtype=int)

    cell_ids = []

    for cell_ind, (cell_id, spike_time_ind) in enumerate(
        spike_times.groupby(["channel", "cluster_ID"])
    ):
        spikes[spike_time_ind, cell_ind] += 1
        cell_ids.append(cell_id)

    return pd.DataFrame(spikes, columns=cell_ids, index=pd.Index(time, name="time"))


def gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    """1D convolution of the data with a Gaussian.
    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`
    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional
    Returns
    -------
    smoothed_data : array_like
    """
    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis, mode="constant"
    )


def get_velocity(position, time=None, sigma=15, sampling_frequency=1):
    if time is None:
        time = np.arange(position.shape[0])

    return gaussian_smooth(
        np.gradient(position, time, axis=0),
        sigma,
        sampling_frequency,
        axis=0,
        truncate=8,
    )


def get_speed(position, time=None, sigma=15, sampling_frequency=1):
    velocity = get_velocity(
        position, time=time, sigma=sigma, sampling_frequency=sampling_frequency
    )
    return np.sqrt(np.sum(velocity**2, axis=1))
