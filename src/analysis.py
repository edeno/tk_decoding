from trajectory_analysis_tools import (
    get_2D_distance,
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
    make_2D_track_graph_from_environment,
    maximum_a_posteriori_estimate,
)


def compute_posterior_statistics(
    position_info, classifier, results, hpd_threshold=0.95
):
    posterior = results.acausal_posterior.sum("state")

    most_probable_decoded_position = maximum_a_posteriori_estimate(posterior)

    track_graph = make_2D_track_graph_from_environment(classifier.environments[0])

    decode_distance_to_animal = get_2D_distance(
        position_info[["x", "y"]].to_numpy(),
        most_probable_decoded_position,
        track_graph=track_graph,
    )

    threshold = get_highest_posterior_threshold(posterior, hpd_threshold)
    hpd_spatial_coverage = get_HPD_spatial_coverage(posterior, threshold)

    return (
        most_probable_decoded_position,
        decode_distance_to_animal,
        hpd_spatial_coverage,
    )
