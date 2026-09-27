"""
Configuration and constants for Business Entity Resolution pipeline.
"""

import os
from pathlib import Path

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "code" / "business_entity_resolution" / "models"

# Ensure output and model directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# File paths
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

MATCHING_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"

# Legal suffixes across US, India, France, and Devanagari transliterations
LEGAL_SUFFIXES = {
    # US / Global
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "l.l.c.", "l.l.c", "llp", "l.l.p.", "l.l.p", "ltd", "limited",
    "lp", "l.p.", "pllc", "gmbh", "pvt", "private", "public",
    # India (English)
    "pvt ltd", "private limited", "p ltd", "pvt. ltd.", "pvt ltd.", "ltd.",
    "enterprises", "enterprise", "traders", "trading", "services",
    "associates", "consultants", "consulting", "solutions", "industries",
    # India (Devanagari)
    "प्राइवेट लिमिटेड", "प्रा. लि.", "प्रा लि", "लिमिटेड", "एलएलपी",
    "इंटरप्राइजेज", "एंटरप्राइजेज", "ट्रेडर्स", "कंसल्टेंसी", "सोल्यूशन्स", "उद्योग",
    # France (open-set country in test)
    "sarl", "s.a.r.l.", "sas", "s.a.s.", "sa", "s.a.", "sci", "s.c.i.",
    "eurl", "e.u.r.l.", "snc", "s.n.c.", "gie", "scop", "sem"
}

# Address abbreviations and expansions
ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "h.no": "houseno",
    "h no": "houseno",
    "no": "number",
    "opp": "opposite",
    "nr": "near",
    "ext": "extension",
    "sec": "sector",
    "ph": "phase",
    "dist": "district",
    "tq": "taluk",
    "po box": "pobox",
    # French address terms
    "r": "rue",
    "bd": "boulevard",
    "av": "avenue",
    "all": "allee",
    "pl": "place",
    "imp": "impasse",
    "rte": "route",
    "faub": "faubourg",
    "bat": "batiment"
}

# Blocking configuration
BLOCKING_CONFIG = {
    "max_candidates_per_s1": 25,
    "min_token_len": 3,
    "max_block_size": 2500,  # drop ubiquitous words to prevent explosion
    "top_k_per_block": 10,
    "char_ngram_n": 3
}

# Model and Decision Parameters (tuned for 200K training sample)
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,       # Slower learning = better generalization
    "num_leaves": 127,           # More capacity for complex name patterns
    "max_depth": 8,
    "min_child_samples": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 800,         # More trees, rely on early stopping
    "random_state": 42,
    "n_jobs": -1
}

# Decision threshold — will be tuned via GridSearch on validation F0.5
# These are starting defaults; actual values come from tune_decision_thresholds()
DECISION_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.18
SINGLETON_THRESHOLD = 0.60
