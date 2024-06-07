import networkx as nx
import numpy as np
import pandas as pd
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import paired_cosine_distances
from track_linearization import make_track_graph as _make_track_graph

from src.parameters import CM_PER_PIXEL, SAMPLING_FREQUENCY


def load_data(
    position_file_name: str,
    spike_file_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads position and spike data from csv files, and computes multiunit firing rate and
    multiunit HSE times.

    Parameters
    ----------
    position_file_name : str
    spike_file_name : str

    Returns
    -------
    position_info : pd.DataFrame
    spikes : pd.DataFrame
    multiunit_firing_rate : pd.DataFrame
    multiunit_HSE_times : pd.DataFrame

    """
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


def get_position_info(file_name: str) -> pd.DataFrame:
    """Loads position information from a csv file, converts to cm, and flips the y-axis,
    and also computes head direction and speed.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    position_info : pd.DataFrame
    """
    position_info = pd.read_csv(file_name)

    # Flip y-axis and convert from pixels to cm
    position_info[["x", "y"]] = (
        flip_y(
            position_info[["x", "y"]].to_numpy(),
            position_info[["x", "y"]].max().to_numpy(),
        )
        * CM_PER_PIXEL
    )

    time = np.arange(len(position_info)) / SAMPLING_FREQUENCY
    position_info = position_info.set_index(pd.Index(time, name="time"))

    velocity = get_velocity(
        position=position_info[["x", "y"]].to_numpy(),
        time=position_info.index.to_numpy(),
        sampling_frequency=SAMPLING_FREQUENCY,
        sigma=0.100,
    )

    position_info["head_direction"] = np.angle(velocity[:, 0] + 1j * velocity[:, 1])
    position_info["speed"] = get_speed(velocity)
    position_info["velocity_x"] = velocity[:, 0]
    position_info["velocity_y"] = velocity[:, 1]

    return position_info


def get_spike_times(file_name: str) -> pd.DataFrame:
    """Loads spike times from a csv file

    Parameters
    ----------
    file_name : str, optional

    Returns
    -------
    spike_times : pd.DataFrame, shape (n_spikes, 1)
    """
    return (
        pd.read_csv(file_name)
        .loc[:, ["channel", "cluster_ID", "photometry_timestamp_250Hz"]]
        .astype(int)
        .set_index(["channel", "cluster_ID"])
        .sort_index()
    )


def convert_spike_times_to_indicator(
    spike_times: pd.DataFrame, time: np.ndarray
) -> pd.DataFrame:
    """Converts spike times to a discrete indicator matrix

    Parameters
    ----------
    spike_times : pd.DataFrame
        Time of each spike, with columns "channel" and "cluster_ID"
    time : np.ndarray, shape (n_time,)
        Time bins of the indicator matrix

    Returns
    -------
    spike_indicator : pd.DataFrame, shape (n_time, n_cells)
        Indicator matrix of spikes
    """
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


def gaussian_smooth(
    data: np.ndarray,
    sigma: float,
    sampling_frequency: int,
    axis: int = 0,
    truncate: int = 8,
) -> np.ndarray:
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


def get_velocity(
    position: np.ndarray, time=None, sigma: float = 0.100, sampling_frequency: int = 1
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    position : np.ndarray, shape (n_time, 2)
    time : np.ndarray, shape (n_time,), optional
    sigma : float, optional
        smoothing parameter, by default 15
    sampling_frequency : int, optional
        samples per second, by default 1

    Returns
    -------
    velocity : np.ndarray, shape (n_time, 2)

    """
    if time is None:
        time = np.arange(position.shape[0])

    return gaussian_smooth(
        np.gradient(position, time, axis=0),
        sigma,
        sampling_frequency,
        axis=0,
        truncate=8,
    )


def get_speed(velocity: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    velocity : np.ndarray, shape (n_time, 2)

    Returns
    -------
    speed : np.ndarray, shape (n_time,)
    """
    return np.sqrt(np.sum(velocity**2, axis=1))


def flip_y(data: np.ndarray, frame_size: np.ndarray) -> np.ndarray:
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


def load_hex_coords(file_name: str) -> pd.DataFrame:
    hex_coords = pd.read_csv(file_name)
    hex_coords[["x", "y"]] = (
        flip_y(
            hex_coords[["x", "y"]].to_numpy(),
            hex_coords[["x", "y"]].max().to_numpy(),
        )
        * CM_PER_PIXEL
    )

    return hex_coords


def hex_occupied(
    hex_coords: pd.DataFrame,
    hex_label: int,
    position_info: pd.DataFrame,
    hex_radius: float = 5.5,
) -> bool:
    hex_center = hex_coords.set_index("hex_label").loc[hex_label]
    return np.any(
        np.linalg.norm(position_info[["x", "y"]] - hex_center[["x", "y"]], axis=1)
        < hex_radius
    )


def make_track_graph(
    position_info: pd.DataFrame, hex_coords: pd.DataFrame, hex_radius: float = 5.5
) -> nx.Graph:
    valid_nodes = [
        int(row.hex_label)
        for row in hex_coords.itertuples(index=False)
        if hex_occupied(row.hex_label, position_info, hex_radius=hex_radius)
    ]

    edges = []
    for hex_label, row in hex_coords.set_index("hex_label").loc[valid_nodes].iterrows():
        for hex_label2, row2 in (
            hex_coords.set_index("hex_label").loc[valid_nodes].iterrows()
        ):
            if hex_label == hex_label2:
                continue
            if np.linalg.norm(row[["x", "y"]] - row2[["x", "y"]]) < hex_radius * 2:
                edges.append((int(hex_label), int(hex_label2)))

    track_graph = _make_track_graph(
        hex_coords[["x", "y"]].to_numpy(), np.array(edges) - 1
    )
    track_graph.remove_nodes_from(list(nx.isolates(track_graph)))

    return track_graph


def determine_if_centrifugal(
    track_graph: nx.Graph,
    track_segment_id: np.ndarray,
    head_direction: np.ndarray,
) -> np.ndarray:
    """
    Determine if movement direction is centrifugal based on head direction similarity.

    Parameters:
    ----------
    track_graph : nx.Graph
        The graph representing the track.
    track_segment_id : np.ndarray, shape (n_time,)
        The IDs of the track segments.
    head_direction : np.ndarray, shape (n_time,)
        The head direction values.

    Returns:
    -------
    is_centrifugal : np.ndarray, shape (n_time,)
        An array indicating whether each movement is centrifugal or not.
    """
    closeness_centrality = nx.centrality.closeness_centrality(
        track_graph, distance="distance"
    )
    centrifugal_edges = [
        (u, v) if closeness_centrality[u] >= closeness_centrality[v] else (v, u)
        for u, v in track_graph.edges()
    ]

    node_positions = nx.get_node_attributes(track_graph, "pos")
    edges = np.array(track_graph.edges())
    edge_nodes = edges[track_segment_id]
    edge_nodes = np.stack((edge_nodes[:, ::-1], edge_nodes), axis=-1)

    dir = np.array(
        [
            (
                np.array(node_positions[node1]) - np.array(node_positions[node2]),
                np.array(node_positions[node2]) - np.array(node_positions[node1]),
            )
            for node1, node2 in edges
        ]
    )[track_segment_id]

    pos_vec = np.stack(
        [
            np.cos(head_direction),
            np.sin(head_direction),
        ],
        axis=1,
    )
    cos_similarity = np.stack(
        (
            1 - paired_cosine_distances(pos_vec, dir[:, 0]),
            1 - paired_cosine_distances(pos_vec, dir[:, 1]),
        ),
        axis=1,
    )
    most_similar_ind = np.argmin(np.abs(1.0 - cos_similarity), axis=1)
    edge_direction = edge_nodes[np.arange(edge_nodes.shape[0]), :, most_similar_ind]
    # Create structured arrays
    dtype = [("node1", edge_direction.dtype), ("node2", edge_direction.dtype)]
    edge_direction = np.array(list(map(tuple, edge_direction)), dtype=dtype)

    is_centrifugal = np.isin(edge_direction, np.array(centrifugal_edges, dtype=dtype))

    return is_centrifugal, centrifugal_edges


def get_auto_linear_edge_order_spacing(
    track_graph: nx.Graph,
) -> tuple[np.ndarray, np.ndarray]:
    linear_edge_order = list(nx.traversal.edge_bfs(track_graph, source=1))
    is_connected_component = ~(
        np.abs(np.array(linear_edge_order)[:-1, 1] - np.array(linear_edge_order)[1:, 0])
        > 0
    )
    linear_edge_spacing = ~is_connected_component * 15.0

    return linear_edge_order, linear_edge_spacing
