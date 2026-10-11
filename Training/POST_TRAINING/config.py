import torch
from pathlib import Path
from typing import Dict, List

# Runtime device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_GPU = False
GPU_IDS: List[int] = []  # [] means use all visible CUDA devices

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "POST_TRAINING"
DATA_DIR = PROJECT_ROOT / "DATA" / "cams_fc_900_451_v1.zarr"
STATS_FILE = PROJECT_ROOT / "DATA" / "Normalization" / "stats_fc_900_451.json"

# Spatial size (width x height)
WIDTH = 900
HEIGHT = 451

# Longitude roll (shift width) to move seam away from 0° meridian.
# Set to WIDTH // 2 to convert 0–360 to -180–180 ordering.
LON_ROLL = WIDTH // 2

# Mask settings (default: full-domain training and evaluation)
EVAL_MASKS: List[str] = []
TRAIN_MASK: str | None = None

# Feature variables per source (from preprocess_data/06make_sample.py)
FEATURE_GROUPS: Dict[str, List[str]] = {
    "cams_ra_surface": [
        "pm1",
        "pm2p5",
        "pm10",
        "tcco",
        "tcno2",
        "tc_no",
        "gtco3",
        "tcso2",
    ],
    "cams_ra_level": [
        "co_1000",
        "no_1000",
        "no2_1000",
        "so2_1000",
        "go3_1000",
    ],
    "era5_surface": ["tcc", "d2m", "t2m","u10m","v10m","msl","tp"],
    "era5_level": [
        "z50","z100","z150","z200","z250","z300","z400","z500","z600","z700","z850","z925","z1000",
        "t50","t100","t150","t200","t250","t300","t400","t500","t600","t700","t850","t925","t1000",
        "u50","u100","u150","u200","u250","u300","u400","u500","u600","u700","u850","u925","u1000",
        "v50","v100","v150","v200","v250","v300","v400","v500","v600","v700","v850","v925","v1000",
    ],
    "emis": [
        "ammonia",
        "black-carbon",
        "carbon-monoxide",
        "nitrogen-oxides",
        "non-methane-vocs",
        "sulphur-dioxide",
    ],
    "temporal": [
        "day_of_year_sin","day_of_year_cos",
        #"day_of_week_sin","day_of_week_cos",
        #"month_of_year_sin","month_of_year_cos",
        "hour_of_day_sin","hour_of_day_cos",
    ],
}

# Time window configuration (offsets are in number of time steps)
TIME_STEP_HOURS = 12
ALLOWED_BASE_HOURS = {0, 12}

FEATURE_TIME_OFFSETS: Dict[str, List[int]] = {
    "cams_ra_surface": [-2, -1],
    "cams_ra_level": [-2, -1],
    "era5_surface": [-1, 0],
    "era5_level": [-1, 0],
    "emis": [-1, 0],
    "temporal": [0],
}
LABEL_TIME_OFFSET: int = 0


def _offset_to_label(offset: int) -> str:
    if offset == 0:
        return "t0"
    if offset < 0:
        return f"t_minus_{abs(offset)}"
    return f"t_plus_{offset}"


FEATURE_TIME_LABELS: Dict[str, List[str]] = {
    group: [_offset_to_label(offset) for offset in offsets]
    for group, offsets in FEATURE_TIME_OFFSETS.items()
}

# Labels: cams_ra_surface/cams_ra_level at configured offset
LABEL_GROUPS: Dict[str, List[str]] = {
    "cams_ra_surface": FEATURE_GROUPS["cams_ra_surface"],
    "cams_ra_level": FEATURE_GROUPS["cams_ra_level"],
}
LABEL_TIME: str = _offset_to_label(LABEL_TIME_OFFSET)
LABEL_VARS: List[str] = LABEL_GROUPS["cams_ra_surface"] + LABEL_GROUPS["cams_ra_level"]

# Normalization
NORMALIZE: bool = True
NORMALIZE_METHOD: str = "zscore"
NORMALIZE_SCOPE: str = "dataset"

# Label normalization
LABEL_NORMALIZE: bool = True
LABEL_NORMALIZE_METHOD: str = "zscore"
LABEL_NORMALIZE_SCOPE: str = "dataset"

# Emission preprocessing (apply before normalization)
EMISSION_PREPROCESS: Dict[str, float | str | None] = {
    "method": "log1p",
    "clip_min": 0.0,
}
EMIS_VARS: List[str] = FEATURE_GROUPS["emis"]

# Clip negatives for selected CAMS variables before normalization.
CLIP_MIN_VALUE: float | None = 0.0
CLIP_MIN_BASE_VARS: List[str] = (
    FEATURE_GROUPS["cams_ra_surface"] + FEATURE_GROUPS["cams_ra_level"]
)

# Training hyperparameters
SEED = 42
BATCH_SIZE = 2
NUM_WORKERS = 16
PREFETCH_FACTOR = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
EPOCHS = 100
WARMUP_STEPS = 1000
AMP_ENABLED = True
AMP_DTYPE = "bf16"
PATIENCE = 30
MIN_DELTA = 0.0
MIN_LR = 1e-8

# Optional pretrained initialization (used only when no latest checkpoint exists).
PRETRAINED_CKPT_PATH: str | None = str(
    PROJECT_ROOT / "PRE_TRAINING" / "checkpoints"
    / "swin_ps4_ed1024_d20_w8_loss_charb1_ssim0p25_eps0p001_2010" / "best_swin_t0.pth"
)

# Loss settings (Charbonnier + SSIM)
LOSS_ALPHA = 1
LOSS_BETA = 0.75
SSIM_DATA_RANGE = "auto"
SSIM_WIN_SIZE = 3
CHARBONNIER_EPSILON = 1e-3
LOSS_CHANNEL_WEIGHT_METHOD: str | None = "none"
LOSS_CHANNEL_WEIGHT_NORMALIZE: bool = True
LOSS_CHANNEL_WEIGHT_EPS: float = 1e-6
LOSS_CHANNEL_WEIGHT_CLIP_MIN: float | None = 0.25
LOSS_CHANNEL_WEIGHT_CLIP_MAX: float | None = 4
LOSS_CHANNEL_WEIGHTS: Dict[str, float] = {}

# Latitude weighting (reduce loss contribution at higher latitudes)
LAT_WEIGHTING: bool = True
LAT_WEIGHT_SOURCE: str | None = None
LAT_WEIGHT_POWER: float = 1.0
LAT_WEIGHT_NORMALIZE: bool = False
LAT_WEIGHT_EPS: float = 1e-6
LAT_WEIGHT_APPLY_TO_SSIM: bool = True

# Swin-Transformer parameters (2D)
SWIN: Dict[str, int] = {
    "patch_size": 4,
    "embed_dim": 1024,
    "depths": [20],
    "num_heads": [16],
    "window_size": 8,
    "time_kernel": 2,
    "dropout": 0.1,
    # Bilinear resize around Swin:
    # input(H,W) -> (swin_resize_height, swin_resize_width) -> Swin -> back to input(H,W)
    "swin_resize_height": 448,
    "swin_resize_width": 896,
}


def _format_list(values: List[int]) -> str:
    return "-".join(str(v) for v in values)


def _format_float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


MODEL_TAG = (
    f"swin_ps{SWIN['patch_size']}"
    f"_ed{SWIN['embed_dim']}"
    f"_d{_format_list(SWIN['depths'])}"
    # f"_h{_format_list(SWIN['num_heads'])}"
    f"_w{SWIN['window_size']}"
)
LOSS_TAG = (
    f"loss_charb{_format_float_tag(LOSS_ALPHA)}"
    f"_ssim{_format_float_tag(LOSS_BETA)}"
    f"_eps{_format_float_tag(CHARBONNIER_EPSILON)}"
    # f"_range{SSIM_DATA_RANGE}"
)
RUN_TAG = f"{MODEL_TAG}_{LOSS_TAG}_2025"

LOG_DIR = CODE_DIR / "logs" / RUN_TAG
CKPT_DIR = CODE_DIR / "checkpoints" / RUN_TAG
RESULT_DIR = CODE_DIR / "result" / RUN_TAG
PLOTS_DIR = RESULT_DIR / "plots"

# Daily split ranges (inclusive). Adjust to your dataset coverage.
TRAIN_START_DATE = "2017-10-01"
TRAIN_END_DATE = "2022-05-31"
VAL_START_DATE = "2022-06-01"
VAL_END_DATE = "2022-11-30"
TEST_START_DATE = "2022-06-01"
TEST_END_DATE = "2022-11-30"

# When set, *_DATE_RANGES overrides *_START_DATE/*_END_DATE.
# Format: [("YYYY-MM-DD", "YYYY-MM-DD"), ...]
TRAIN_DATE_RANGES = None
VAL_DATE_RANGES = None
TEST_DATE_RANGES = None
