import numpy as np
from trajectory_analysis_tools import (
    get_2D_distance,
    get_highest_posterior_threshold,
    make_2D_track_graph_from_environment,
    maximum_a_posteriori_estimate,
)


def compute_posterior_statistics(position_info, classifier, results, hpd_coverage=0.95):
    posterior = results.acausal_posterior.sum("state")

    most_probable_decoded_position = maximum_a_posteriori_estimate(posterior)

    track_graph = make_2D_track_graph_from_environment(classifier.environments[0])

    decode_distance_to_animal = get_2D_distance(
        position_info[["x", "y"]].to_numpy(),
        most_probable_decoded_position,
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
    hpd_spatial_coverage = (isin_hpd * bin_area).sum("position")

    return (
        most_probable_decoded_position,
        decode_distance_to_animal,
        hpd_spatial_coverage,
    )
