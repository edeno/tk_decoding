"""Run decoding for a given animal and date."""
import glob
import logging
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import cupy as cp
import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from replay_trajectory_classification import (
    Environment,
    RandomWalk,
    SortedSpikesClassifier,
    Uniform,
)

from src.analysis import compute_posterior_statistics
from src.load_data import load_data
from src.parameters import PROCESSED_DATA_DIR, RAW_DATA_DIR, SAMPLING_FREQUENCY
from src.plot_data import create_interactive_2D_decoding_figurl


def setup_logger(name_logfile: str, path_logfile: str) -> logging.Logger:
    """Sets up a logger for each function that outputs
    to the console and to a file

    Parameters
    ----------
    name_logfile : str
        Name of the logfile
    path_logfile : str
        Path to the logfile

    Returns
    -------
    logger : logging.Logger
        Logger object
    """
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    )
    fileHandler = logging.FileHandler(path_logfile, mode="w")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger


def run_decode(
    animal: str,
    date: str,
    create_figurl: bool = True,
    log_directory="",
    overwrite=True,
):
    """Run decoding for a given animal and date.

    Parameters
    ----------
    animal : str
        Name of the animal, e.g. 'JZ1'
    date : str
        Date animal was run, in format YYYY-MM-DD
    create_figurl : bool, optional
        Make an interactive visualization and store in results, by default True
    log_directory : str, optional
        Directory to store log files, by default ""
    overwrite : bool, optional
        Overwrite existing results, by default True
    """

    try:
        file_name = f"{animal}_{date}"
        logger = setup_logger(
            name_logfile=file_name,
            path_logfile=os.path.join(log_directory, f"{file_name}.log"),
        )

        logger.info(" START ".center(50, "#"))
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            universal_newlines=True,
        ).stdout
        logger.info("Git Hash: {git_hash}".format(git_hash=git_hash.rstrip()))

        # Check if results exist
        results_filename = os.path.join(PROCESSED_DATA_DIR, f"{file_name}_results.nc")

        if not Path(results_filename).is_file() or overwrite:
            # Garbage collect GPU memory
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

            logger.info("Loading data...")

            position_file_name = os.path.join(
                RAW_DATA_DIR, animal, date, f"{animal}_{date}_position.csv"
            )
            spike_file_name = os.path.join(
                RAW_DATA_DIR, animal, date, f"{animal}_{date}_spikesWithPosition.csv"
            )
            (
                position_info,
                spikes,
                multiunit_firing_rate,
                multiunit_HSE_times,
            ) = load_data(
                position_file_name=position_file_name,
                spike_file_name=spike_file_name,
            )
            logger.info("Finished loading data...")

            # cut out first 20 s because animal is being placed on track
            start_ind = int(20.0 * SAMPLING_FREQUENCY)
            position_info = position_info.iloc[start_ind:]
            spikes = spikes.iloc[start_ind:]
            multiunit_firing_rate = multiunit_firing_rate.iloc[start_ind:]

            # Save out data
            position_info.to_csv(
                os.path.join(PROCESSED_DATA_DIR, f"{animal}_{date}_position_info.csv")
            )
            spikes.to_csv(
                os.path.join(PROCESSED_DATA_DIR, f"{animal}_{date}_spikes.csv")
            )
            multiunit_firing_rate.to_csv(
                os.path.join(
                    PROCESSED_DATA_DIR, f"{animal}_{date}_multiunit_firing_rate.csv"
                )
            )
            multiunit_HSE_times.to_csv(
                os.path.join(
                    PROCESSED_DATA_DIR,
                    f"{animal}_{date}_multiunit_HSE_times.csv",
                )
            )

            state_names = ["continuous", "fragmented"]
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
                time_slice = slice(
                    ind * n_time // n_segments, (ind + 1) * n_time // n_segments
                )
                results.append(
                    classifier.predict(
                        spikes.iloc[time_slice],
                        time=spikes.iloc[time_slice].index.to_numpy(),
                        state_names=state_names,
                        use_gpu=True,
                    )
                )

            results = xr.concat(results, dim="time")
            logger.info("Finished decoding...")

            logger.info("Computing statistics...")
            (
                most_probable_decoded_position,
                decode_distance_to_animal,
                hpd_spatial_coverage,
            ) = compute_posterior_statistics(
                position_info, classifier, results, hpd_coverage=0.95
            )
            results["most_probable_decoded_position"] = (
                ["time", "position"],
                most_probable_decoded_position,
            )
            results["decode_distance_to_animal"] = ("time", decode_distance_to_animal)
            results["hpd_spatial_coverage"] = ("time", np.asarray(hpd_spatial_coverage))
            logger.info("Finished computing statistics...")

            if create_figurl:
                logger.info("Creating figurls...")
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
                        view_height=800,
                    )
                    attrs["figurl_{ind}"] = view.url(label=f"{animal}_{date}_{ind}")

                results = results.assign_attrs(attrs)
                logger.info("Finished creating figurls...")

            results.drop(["likelihood", "causal_posterior"]).to_netcdf(results_filename)
            classifier.save_model(
                os.path.join(PROCESSED_DATA_DIR, f"{animal}_{date}_classifier.pkl")
            )

            logger.info("Done!")
    except Exception as e:
        logger.warning(e)


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--cuda_visible_devices", type=str, default="0,1,2,3,4,5,6,7,8,9"
    )
    parser.add_argument("--local_scratch_dir", type=str, default=".local_scratch")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--create_figurl", action="store_true")
    parser.add_argument("--animal", type=str)
    parser.add_argument("--date", type=int)

    return parser.parse_args()


def get_session_info():
    """Get information about the current session (e.g. the animal and date)"""
    raw_data_files = [
        os.path.basename(os.path.normpath(file_name)).split("_")[:2]
        for file_name in glob.glob(
            os.path.join(RAW_DATA_DIR, "**/*position.csv"), recursive=True
        )
    ]
    return (
        pd.DataFrame(raw_data_files, columns=["animal", "date"])
        .astype({"date": int})
        .set_index(["animal", "date"])
    )


if __name__ == "__main__":
    args = get_command_line_arguments()

    os.makedirs(args.local_scratch_dir, exist_ok=True)

    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=args.cuda_visible_devices,
        local_directory=args.local_scratch_dir,
    )
    client = Client(cluster)

    log_directory = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_directory, exist_ok=True)

    session_info = get_session_info()

    if args.animal is not None:
        epoch_key = args.animal, args.date
        session_info = session_info.xs(epoch_key, drop_level=False)
        if isinstance(session_info, pd.Series):
            session_info = session_info.to_frame().T

    # Append the result of the computation into a results list
    results = [
        dask.delayed(run_decode)(
            animal=animal,
            date=date,
            create_figurl=args.create_figurl,
            log_directory=log_directory,
            overwrite=args.overwrite,
        )
        for animal, date in session_info.index
    ]

    # Run `dask.compute` on the results list for the code to run
    dask.compute(*results)
