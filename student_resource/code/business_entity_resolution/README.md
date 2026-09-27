# Business Entity Resolution Pipeline — ML Challenge 2026

An end-to-end Entity Resolution pipeline designed to match noisy, heterogeneous business records across multiple sources while optimizing for **Macro $F_{0.5}$** (precision-weighted evaluation) and preventing false merges.

---

## 1. Project Structure

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── matching_results.tsv        # Final entity matches (submitted to leaderboard)
│   └── candidate_pairs.tsv         # Candidate pairs from multi-tier blocking
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt
│       ├── README.md
│       └── src/
│           ├── __init__.py
│           ├── config.py           # Paths, constants, parameters
│           ├── normalizer.py       # Indic/Latin accent cleaning, legal suffix removal
│           ├── data_loader.py      # Streaming TSV parsing & I/O
│           ├── blocking.py         # Multi-tier inverted index candidate generator
│           ├── feature_extraction.py # RapidFuzz pairwise string & address features
│           ├── dataset_builder.py  # Hard negative sampling & matrix construction
│           ├── model.py            # GBDT classifier wrapper (LightGBM/XGBoost)
│           ├── evaluation.py       # Macro F0.5, Precision, Recall, Singleton accuracy
│           └── pipeline.py         # End-to-end driver
├── utils/
│   └── validate_submission.py
└── Documentation_template.md
```

---

## 2. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Reproduction & Execution

### Run Complete End-to-End Pipeline (Training + Validation + Test Inference + Validation)

From `student_resource/`:

```bash
python -m code.business_entity_resolution.src.pipeline --sample-size 150000 --folds 3
```

This will automatically:
1. Parse and pre-normalize training & test records.
2. Build multi-tier inverted indices and generate candidates.
3. Compute pairwise string/token/address similarity features.
4. Perform 3-fold GroupKFold cross-validation, tuning decision thresholds for Macro $F_{0.5}$.
5. Fit the final GBDT model and compute test predictions.
6. Export `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
7. Validate the output format using `utils/validate_submission.py`.

### Run Test Inference Only (Using Saved Model)

```bash
python -m code.business_entity_resolution.src.pipeline --inference-only
```

---

## 4. Submission Validation

Verify output formatting independently:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
