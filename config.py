"""
config.py – Central configuration for VictorIA.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ENSEMBLE_PATH = CACHE_DIR / "ensemble.pkl"
SCALER_PATH = CACHE_DIR / "scaler.pkl"
METRICS_PATH = CACHE_DIR / "metrics.json"

# Training
TRAINING_SAMPLES = 2000
TRAINING_TEST_SIZE = 0.2
TRAINING_RANDOM_STATE = 42

# Ensemble weights [XGBoost, RandomForest, LightGBM, NeuralNet]
ENSEMBLE_WEIGHTS = [0.40, 0.25, 0.35, 0.0]  # NN disabled by default (no TensorFlow)
ENSEMBLE_FALLBACK_WEIGHTS = [0.5, 0.5, 0.0, 0.0]
ENSEMBLE_IDX_XGB = 0
ENSEMBLE_IDX_RF = 1
ENSEMBLE_IDX_LGBM = 2
ENSEMBLE_IDX_NN = 3

# Bump when logic changes to invalidate Streamlit cache
APP_VERSION = "1.4.0"

# Blend ML output with empirical stats when real API data is available
EMPIRICAL_BLEND_WEIGHT = 0.75

# Real historical training (football-data.org)
COMPETITION_CODES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Ligue 1": "FL1",
    "Serie A": "SA",
    "Bundesliga": "BL1",
}
TOP5_COMPETITION_CODES = ["PL", "PD", "FL1", "SA", "BL1"]
REAL_DATASET_PATH = CACHE_DIR / "real_dataset.pkl"
REAL_DATASET_META_PATH = CACHE_DIR / "real_dataset_meta.json"
DEFAULT_TRAINING_SEASONS = 4  # number of past seasons to fetch
MIN_TEAM_MATCHES = 3
STATS_WINDOW = 10
API_REQUEST_DELAY_SEC = 6.5  # free tier ~10 req/min

# Advanced feature engineering
RECENT_FORM_WEIGHT = 0.65
HISTORICAL_FORM_WEIGHT = 0.35
NIGHT_MATCH_START_HOUR = 18
MAX_VALID_GOALS_PER_TEAM = 12
MAX_POINTS_PER_MATCH = 3
MIN_ADAPTIVE_WINDOW = 5
ADAPTIVE_HISTORY_RATIO = 0.7
TEAM_STRENGTH_WEIGHT_WIN_RATE = 0.5
TEAM_STRENGTH_WEIGHT_FORM = 0.3
TEAM_STRENGTH_WEIGHT_GOAL_EDGE = 0.2
TEAM_STRENGTH_FORM_DIVISOR = 3.0

# Validation
TIME_SERIES_SPLITS = 5
STRATIFIED_FOLDS = 5
MIN_SAMPLES_PER_TIMESERIES_FOLD = 20
