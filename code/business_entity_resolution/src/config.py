"""
Configuration parameters for Entity Resolution Pipeline
Amazon ML Challenge 2026
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "student_resource" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

# Ensure output and model directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_OUTPUT_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Training configuration
TRAIN_SAMPLE_PER_COUNTRY = 30000  # High-quality balanced training sample
MAX_CANDIDATES_PER_ENTITY = 15     # Compact, high-scoring candidate budget
CLASSIFICATION_THRESHOLD = 0.55    # Calibrated for Macro F_0.5

# LightGBM Hyperparameters
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42
}
NUM_BOOST_ROUND = 350
