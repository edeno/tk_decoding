import numpy as np
import pandas as pd
import xarray as xr
from replay_trajectory_classification import SortedSpikesClassifier
from trajectory_analysis_tools import (
    get_2D_distance,
    get_ahead_behind_distance2D,
    get_highest_posterior_threshold,
    make_2D_track_graph_from_environment,
    maximum_a_posteriori_estimate,
)


def compute_posterior_statistics(
    position_info: pd.DataFrame,
    classifier: SortedSpikesClassifier,
    results: xr.Dataset,
    hpd_coverage: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute statistics from the decoding results.

    Parameters
    ----------
    position_info : pd.DataFrame
    classifier : SortedSpikesClassifier
    results : xr.Dataset
    hpd_coverage : float, optional

    Returns
    -------
    most_probable_decoded_position : np.ndarray, shape (n_time, 2)
    decode_distance_to_animal : np.ndarray, shape (n_time,)
    hpd_spatial_coverage : np.ndarray, shape (n_time,)

    """
    posterior = results.acausal_posterior.sum("state")

    most_probable_decoded_position = maximum_a_posteriori_estimate(posterior)

    track_graph = make_2D_track_graph_from_environment(classifier.environments[0])

    ahead_behind_distance = get_ahead_behind_distance2D(
        head_position=position_info[["x", "y"]].to_numpy(),
        head_direction=position_info["head_direction"].to_numpy(),
        map_position=np.asarray(most_probable_decoded_position),
        track_graph=track_graph,
    )

    hpd_threshold = get_highest_posterior_threshold(posterior, hpd_coverage)
    stacked_posterior = posterior.stack(position=["x_position", "y_position"]).dropna(
        "position"
    )
    bin_area = np.median(np.diff(posterior.x_position)) * np.median(
        np.diff(posterior.y_position)
    )
    isin_hpd = stacked_posterior >= hpd_threshold[:, np.newaxis]
    hpd_spatial_coverage = np.asarray((isin_hpd * bin_area).sum("position"))

    return (
        most_probable_decoded_position,
        ahead_behind_distance,
        hpd_spatial_coverage,
    )
