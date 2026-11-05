"""Constants of the probe.

Everything marked "from OpenTSLM" must match the upstream code byte for byte:
the probe is only meaningful if the unperturbed condition reproduces the input
the model was trained on.
"""

SEED = 42

# Upstream commit the probe was written against (github.com/StanfordBDHG/OpenTSLM).
OPENTSLM_COMMIT = "2968f4b"

# Released HAR-CoT data, the same URL OpenTSLM's loader downloads from.
HAR_COT_URL = "https://polybox.ethz.ch/index.php/s/kD74GnMYxn3HBEM/download"

SPLIT_FILES = {
    "train": "har_cot_train_cot.csv",
    "validation": "har_cot_val_cot.csv",
    "test": "har_cot_test_cot.csv",
}

AXES = ("x_axis", "y_axis", "z_axis")
SEQ_LEN = 128  # 2.56 s at 50 Hz, fixed by the dataset

# From OpenTSLM: HARCoTQADataset.get_labels()
LABELS = (
    "biking",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
    "walking_down",
    "walking_up",
)

# From OpenTSLM: har_cot/HARCoTQADataset.py, TIME_SERIES_LABELS
TIME_SERIES_LABELS = (
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
)

# From OpenTSLM: the clamp applied to each axis' std before normalising. The
# clamped value is also the one printed in the prompt.
MIN_STD = 1e-6
