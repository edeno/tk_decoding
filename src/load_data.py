import numpy as np
import pandas as pd
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)
from scipy.ndimage import gaussian_filter1d

from src.parameters import CM_PER_PIXEL, SAMPLING_FREQUENCY


def load_data(
    position_file_name="../Raw-Data/position4Xulu.csv",
    spike_file_name="../Raw-Data/df4Xulu.csv",
):
    position_info = get_position_info(position_file_name)
    spike_times = get_spike_times(spike_file_name)

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


def get_position_info(file_name="../Raw-Data/position4Xulu.csv"):
    position_info = pd.read_csv(file_name)[["x", "y"]]

    # Flip y-axis and convert from pixels to cm
    position_info = pd.DataFrame(
        flip_y(position_info.to_numpy(), position_info.max().to_numpy()) * CM_PER_PIXEL,
        columns=position_info.columns,
        index=position_info.index,
    )

    time = np.arange(len(position_info)) / SAMPLING_FREQUENCY
    position_info = position_info.set_index(pd.Index(time, name="time"))

    position_info["speed"] = get_speed(
        position=position_info.to_numpy(),
        time=position_info.index.to_numpy(),
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    return position_info


def get_spike_times(file_name="../Raw-Data/df4Xulu.csv"):
    return (
        pd.read_csv(file_name)
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


def get_speed(position, time=None, sigma=0.100, sampling_frequency=1):
    velocity = get_velocity(
        position, time=time, sigma=sigma, sampling_frequency=sampling_frequency
    )
    return np.sqrt(np.sum(velocity**2, axis=1))


def flip_y(data, frame_size):
    """Flips the y-axis
    Parameters
    ----------
    data : ndarray, shape (n_time, 2)
    frame_size : array_like, shape (2,)
    Returns
    -------
    flipped_data : ndarray, shape (n_time, 2)
    """
    new_data = data.copy()
    if data.ndim > 1:
        new_data[:, 1] = frame_size[1] - new_data[:, 1]
    else:
        new_data[1] = frame_size[1] - new_data[1]
    return new_data
