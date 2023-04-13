import os

SAMPLING_FREQUENCY = 250  # samples per second
CM_PER_PIXEL = 1 / 3.14  # cm / pixels


# Data directories and definitions
ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "Processed-Data")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "Raw-Data")
