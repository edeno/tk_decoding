import logging

logging.basicConfig(
    level="INFO", format="%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
from src.analysis import compute_posterior_statistics
from src.load_data import load_data
from src.parameters import SAMPLING_FREQUENCY
from replay_trajectory_classification import (
    SortedSpikesClassifier,
    Environment,
    RandomWalk,
    Uniform,
)
from src.plot_data import create_interactive_2D_decoding_figurl

import cupy as cp
import xarray as xr


def run_decode(create_figurl=True):
    state_names = ["continuous", "fragmented"]

    logging.info("Loading data...")
    (position_info, spikes, multiunit_firing_rate, multiunit_HSE_times) = load_data(
        position_file_name="../Raw-Data/position4Xulu.csv",
        spike_file_name="../Raw-Data/df4Xulu.csv",
    )
    logging.info("Finished loading data...")

    # cut out first 20 s because animal is being placed on track
    start_ind = int(20.0 * SAMPLING_FREQUENCY)
    position_info = position_info.iloc[start_ind:]
    spikes = spikes.iloc[start_ind:]
    multiunit_firing_rate = multiunit_firing_rate.iloc[start_ind:]

    environment = Environment(place_bin_size=2.0)
    continuous_transition_types = [
        [RandomWalk(movement_var=12.0), Uniform()],
        [Uniform(), Uniform()],
    ]

    classifier = SortedSpikesClassifier(
        environments=environment,
        continuous_transition_types=continuous_transition_types,
        sorted_spikes_algorithm="spiking_likelihood_kde_gpu",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "use_diffusion": False,
            "block_size": int(2**12),
        },
    )

    n_time = len(spikes)
    n_segments = n_time // (60 * 60 * SAMPLING_FREQUENCY)  # 1 hour segments
    results = []

    # Fit the place fields
    classifier.fit(
        position=position_info[["x", "y"]],
        spikes=spikes,
    )

    for ind in range(n_segments):
        time_slice = slice(ind * n_time // n_segments, (ind + 1) * n_time // n_segments)
        results.append(
            classifier.predict(
                spikes.iloc[time_slice],
                time=spikes.iloc[time_slice].index.to_numpy(),
                state_names=state_names,
                use_gpu=True,
            )
        )

    results = xr.concat(results, dim="time")
    logging.info("Finished decoding...")

    logging.info("Computing statistics...")
    (
        most_probable_decoded_position,
        decode_distance_to_animal,
        hpd_spatial_coverage,
    ) = compute_posterior_statistics(
        position_info, classifier, results, hpd_coverage=0.95
    )
    results["most_probable_decoded_position"] = ("time", most_probable_decoded_position)
    results["decode_distance_to_animal"] = ("time", decode_distance_to_animal)
    results["hpd_spatial_coverage"] = ("time", hpd_spatial_coverage)
    logging.info("Finished computing statistics...")

    if create_figurl:
        logging.info("Creating figurls...")
        attrs = dict()
        for ind in range(n_segments):
            time_slice = slice(
                ind * n_time // n_segments, (ind + 1) * n_time // n_segments
            )

            view = create_interactive_2D_decoding_figurl(
                position_info.iloc[time_slice],
                multiunit_firing_rate.iloc[time_slice],
                results.isel(time=time_slice),
                bin_size=environment.place_bin_size,
                position_name=["x", "y"],
                speed_name="speed",
                posterior_type="acausal_posterior",
                view_height=800,
            )
            attrs["figurl_{ind}"] = view.url(label=f"Barrier 2D Decode_{ind}")

        results = results.assign_attrs(attrs)
        logging.info("Finished creating figurls...")

    results.drop(["likelihood", "causal_posterior"]).to_netcdf(
        "../Processed-Data/results_0.nc"
    )
    classifier.save_model("../Processed-Data/classifier_0.pkl")

    logging.info("Done!")
