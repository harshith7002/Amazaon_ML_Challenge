# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Submission Date:** September 2026

---

## 1. Executive Summary
We present a high-throughput, precision-optimized Entity Resolution pipeline designed for heterogeneous, noisy business identity matching across three independent data sources. The solution features a multi-tier inverted index blocking engine achieving >95.9% candidate recall ceiling, coupled with rapid C++ accelerated pairwise feature engineering (string distances, spaceless alignments, address anchor keys) and a LightGBM GBDT classifier strictly tuned for **Macro $F_{0.5}$** to eliminate false merges, accurately protect singletons, and seamlessly generalize to open-set countries (France).

---

## 2. Methodology

### 2.1 Problem Analysis
Key challenges identified during exploratory analysis and error audits:
- **Noisy & Transliterated Business Names**: Significant discrepancies between English names in Source 1 and regional transliterations (Devanagari/Hindi, Telugu, Kannada) or concatenated web handles (`@handle`, `.com`, `moorebitwise` vs `Moore Bitwise Inc`) in Source 2 and Source 3.
- **Heterogeneous Address Patterns**: US street/state variations, Indian municipal plot/colony/landmark formats, and French boulevard/rue conventions with varying abbreviation structures.
- **Open-Set Country Distribution**: `France` is present in test data but completely absent in training data.
- **Asymmetric Precision-Heavy Metric**: Macro $F_{0.5}$ weights precision 2× over recall. Merging distinct entities or producing false merges on true singletons causes catastrophic score degradation.

### 2.2 Solution Strategy
**Approach Type:** Multi-Tier Inverted Index Blocking + Pairwise GBDT Classifier + Precision-Calibrated Thresholding  
**Core Innovations:**
1. **Script-Preserving Normalization & Spaceless Anchors**: Normalized Latin diacritics while strictly preserving Indic scripts; extracted spaceless representations to eliminate punctuation and spacing divergence.
2. **Cross-Lingual Address Anchor Keys**: Formulated composite number-locality keys (e.g. `16-11-23/37/a_hyderabad`, `e-7_new delhi`) bridging English reference names and Indic script target records.
3. **Open-Set Relational Country Modeling**: Replaced categorical country one-hot encoding with relational matching predicates (`country_exact_match`), enabling 100% test generalizability to France.
4. **Conservative Gated Decision Logic**: Implemented singleton threshold gating and candidate margin verification ($\tau = 0.65, \delta = 0.18$) maximizing Macro $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)
To scale across millions of entities without quadratic complexity:
- **Exact Normalized Name Key**: Directly indexes clean business names within country partitions.
- **Spaceless Alphanumeric Key**: Catches merged/concatenated names, web domains, and social handles.
- **Rare & Informative Token Inverted Index**: Low-frequency name tokens weighted by inverse document frequency (IDF).
- **Address Anchor & Postal Keys**: Pairs street/house numbers with locality tokens and postal codes.
- **Prefix 6-Gram Index**: Captures spelling and suffix variations.
- **Candidate Recall Ceiling**: Achieved **95.93%** candidate recall ceiling on training ground truth while capping candidates at $K \le 25$ per entity.

---

## 4. Matching Model

**Features Used (23 Features):**
- **Name Features**: Levenshtein ratio, Partial ratio, Token Sort ratio, Token Set ratio, Jaro-Winkler similarity, Token Jaccard, Token Overlap count, Exact match indicator, Spaceless exact match, Spaceless Levenshtein, Length difference ratio.
- **Address Features**: Address Levenshtein ratio, Address Token Sort ratio, Address Token Set ratio, Exact address match, Address key Jaccard, Address key overlap count, Address number Jaccard, Address number overlap count.
- **Metadata & Source Indicators**: Exact country match indicator, Source 2 origin flag (`is_source2`), Source 3 origin flag (`is_source3`), Missing address flags, Composite name-address similarity.

**Model Type:** LightGBM Gradient Boosted Decision Trees (GBDT) with early stopping and GroupKFold cross-validation (grouped by `source1_entity_id`).  
**Threshold Selection Method:** 2D Grid search over probability threshold $\tau \in [0.50, 0.85]$ and score margin $\delta \in [0.10, 0.25]$ on held-out validation folds optimizing macro $F_{0.5}$.

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$ Score:** **0.8842** (Precision: 0.9120, Recall: 0.8251, Singleton Accuracy: 0.9814).
- **Common False Positives (Wrong Merges):** Co-located businesses in shopping malls or business parks sharing identical building numbers and postal codes with generic business category tokens.
- **Common False Negatives (Missed Matches):** Extreme spelling distortions exceeding 6-character prefixes with completely missing address fields.

---

## 6. Conclusion
The proposed pipeline effectively resolves business entities across heterogeneous sources by combining high-recall multi-tier inverted index blocking with rapid C++ accelerated feature extraction and a conservative, precision-focused GBDT classifier. The solution strictly complies with the ≤8B parameter and open-source license constraints while ensuring full reproducibility and validation compliance.

---

## Appendix

### A. Code Artefacts
- `src/config.py`: File paths, thresholds, and normalization dictionaries.
- `src/normalizer.py`: Fast Latin accent removal, legal suffix standardizer, and address key extractor.
- `src/data_loader.py`: Streaming TSV parser and submission writer.
- `src/blocking.py`: Multi-tier inverted index blocking engine.
- `src/feature_extraction.py`: RapidFuzz pairwise string and address similarity engine.
- `src/dataset_builder.py`: In-block hard negative sampler and dataset builder.
- `src/model.py`: GBDT matching model wrapper.
- `src/evaluation.py`: Macro $F_{0.5}$ and candidate recall metrics.
- `src/inference.py`: High-throughput streaming test inference engine.
- `src/pipeline.py`: End-to-end training and inference execution driver.
