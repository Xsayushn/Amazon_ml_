# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Data Titans  
**Team Members:** Hackathon Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

This solution presents a scalable, production-grade Entity Resolution pipeline tailored for Amazon's multi-source Business Entity Resolution Challenge across 11.7 million test records. Our approach combines **strict country-isolated partitioning**, a **multilingual phonetic transliteration engine** that bridges English and native Indic scripts, an **adaptive multi-pass inverted index blocking mechanism**, and a **LightGBM gradient-boosted pairwise classifier** calibrated specifically for the precision-heavy **Macro $F_{0.5}$** metric. On held-out validation data, our pipeline achieves an exceptional **Macro $F_{0.5}$ score of 0.9613 (96.13%)** while maintaining an ultra-compact candidate pool of **~15 candidates per Source 1 entity**, directly satisfying Amazon's core evaluation requirement of scalable candidate reduction.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) across 24.2 million records revealed several fundamental structural and noise patterns:
1. **Zero Cross-Country Matching**: Rigorous verification against the ground truth demonstrated 0.0% cross-country links. A business located in the US never matches an entity in India or France.
2. **Multilingual Script Shift**: While Source 1 records are 100% Latin (English/Romanized), Source 2 and Source 3 contain native Indic scripts (Devanagari, Tamil, Telugu, Gujarati, Bengali, Kannada, Malayalam) representing phonetic transliterations of Indian businesses (e.g., *'ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி'* matching *'Raj Investments LLP'*, and *'एसएस फूड प्राइवेट लिमिटेड'* matching *'Ss Food Private Limited'*).
3. **Severe Address Asymmetry & Missingness**: In Source 1, 0.0% of addresses are missing. However, ~3.32% of Source 2 and 3 records have completely empty addresses, necessitating a dual-signal matching strategy where records without addresses rely on robust phonetic and character n-gram name matching, while records with DBA/trade names rely on address overlap.
4. **Subtle Name Variations**: Name differences encompass corporate legal suffix substitutions (Inc, LLC, Corp, Pvt Ltd, LLP, SARL), domain-name disguises (e.g. *maurewilliamscolombier.com* for *Maure Williams Colombier Inc*), and abbreviations.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Pass Blocking + Pairwise GBDT Classifier + Macro $F_{0.5}$ Threshold Calibration.  
**Core Innovation:**
- **Cross-Lingual Phonetic Normalization**: Unidecode transliteration coupled with consecutive repeated-character collapsing and Indic phonetic dictionaries maps native Indian scripts to exact Latin matches with 100% token agreement.
- **Overlap-Aware Address Matching**: Implementation of the Szymkiewicz–Simpson overlap coefficient ($\frac{|A \cap B|}{\min(|A|, |B|)}$) specifically captures partial addresses and landmark annotations without penalizing long-versus-short address variations.
- **Precision-Calibrated Singleton Protection**: Because $F_{0.5}$ penalizes false positives twice as heavily as false negatives and singletons represent 5.58% of entities, threshold tuning directly optimizes macro-averaged $F_{0.5}$, yielding 1.0 scores on true singletons.

---

## 3. Candidate Generation (Blocking)

To comply with Amazon's mandate that *"blocking has to scale"* and *"the approach that generates a smaller candidate set per Source 1 entity will be ranked higher"*, our blocking stage employs a 5-pass inverted index with adaptive frequency pruning:

- **Blocking keys used:**
  1. *Core Name Tokens*: Informative business tokens (length $\ge 3$) excluding generic stopwords and legal suffixes.
  2. *Compressed Name Prefixes*: 5-character and 7-character alphanumeric prefixes of the normalized business name.
  3. *Postal / PIN Codes*: 5-digit US ZIP, 5-digit French code postal, and 6-digit Indian PIN codes.
  4. *Distinctive Address Tokens*: Significant words in the address (length $\ge 6$).
  5. *Numeric Building / Shop Identifiers*: Digits of length $\ge 3$.
- **Frequency Pruning**: Keys appearing in $>800$ records are automatically pruned to prevent dense hubs.
- **Candidate pairs generated:** Capped at **$\le 15$ candidates** per Source 1 entity (average ~13.4 to 15.0 candidates).
- **Candidate Recall Ceiling**: **94.67% true match recall** maintained despite the ultra-compact candidate budget.

---

## 4. Matching Model

**Features used (20 total):**
- *Name Similarity (8 features)*:
  - `name_token_sort_ratio`: Word-order invariant Levenshtein ratio.
  - `name_token_set_ratio`: Subset/superset token ratio.
  - `name_core_sort`: Similarity on core name stripped of legal suffixes.
  - `name_ratio`: Raw character-level Levenshtein similarity.
  - `comp_exact` & `comp_ratio`: Exact binary match and ratio on alphanumeric compressed strings.
  - `name_jaccard` & `name_overlap`: Token Jaccard index and Szymkiewicz–Simpson overlap coefficient.
- *Address Similarity (6 features)*:
  - `addr_token_sort_ratio` & `addr_token_set_ratio`.
  - `addr_jaccard` & `addr_overlap`.
  - `postal_match`: Exact match (1.0), mismatch (0.0), or absent (-1.0).
  - `digit_overlap`: Jaccard similarity of street and unit numbers.
- *Structural & Cross-Field (6 features)*:
  - `addr_empty`: Indicator for records lacking address information.
  - `src_type`: Categorical indicator (Source 2 vs Source 3).
  - `shared_keys`: Number of blocking keys shared by the pair.
  - `max_sim`, `min_sim`, `mean_sim`: Cross-field aggregated similarity bounds.

**Model type:** LightGBM Gradient Boosted Decision Tree (GBDT) with 350 estimators, max depth 6, and 31 leaves.  
**Threshold selection method:** Grid search directly maximizing Macro $F_{0.5}$ on held-out validation entities. Calibrated optimal threshold: **0.55**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9613 (96.13%)** on held-out validation split.
- **Candidate Generation Ceiling:** **94.67%** recall with only **15.0** average candidates per entity.
- **Common False Positives (Wrong Merges):**
  - Chain stores sharing identical brand names in the same city with slightly ambiguous branch addresses.
  - Addressed via postal code verification and strict probability thresholding ($\tau = 0.55$).
- **Common False Negatives (Missed Matches):**
  - Records where both the business name is a distinct trade alias (DBA) AND the address is completely omitted.

---

## 6. Conclusion

Our end-to-end entity resolution pipeline delivers top-tier performance on the Amazon ML Challenge 2026. By solving the cross-lingual Indic script shift, enforcing clean country partitioning, constructing a high-recall yet ultra-compact candidate blocking index, and optimizing a GBDT classifier specifically for the precision-heavy Macro $F_{0.5}$ metric, we provide a solution that is both academically rigorous and production-viable at Amazon scale.

---

## Appendix

### A. Code Artefacts
The complete runnable source code is organized under `code/business_entity_resolution/`:
- `src/config.py`: File paths, hyperparameter configuration, and country partition definitions.
- `src/normalizer.py`: Multilingual Indic transliteration and address parser.
- `src/blocking.py`: Multi-pass inverted index with frequency pruning.
- `src/features.py`: 20-dimensional pairwise feature extraction.
- `src/classifier.py`: LightGBM model training, persistence, and macro $F_{0.5}$ metric optimizer.
- `src/pipeline.py`: Full end-to-end execution script.
- `requirements.txt`: Pinned dependencies.
- `README.md`: Step-by-step reproduction instructions.

Entry point:
```bash
python code/business_entity_resolution/src/pipeline.py
```

### B. Additional Results
| Component | Metric | Value |
| :--- | :--- | :--- |
| Candidate Generation | Candidate Recall Ceiling | **94.67%** |
| Candidate Generation | Avg Candidates / S1 Entity | **15.0** |
| Matching Model | Validation Precision | **96.84%** |
| Matching Model | Validation Recall | **93.36%** |
| Overall Evaluation | **Validation Macro $F_{0.5}$** | **0.9613** |
