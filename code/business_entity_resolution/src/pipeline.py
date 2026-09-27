"""
End-to-End Pipeline for Business Entity Resolution
Amazon ML Challenge 2026

Runs:
1. Training on Ground Truth (US + India)
2. Blocking & Candidate Generation on Test Set (US, India, France) -> output/candidate_pairs.tsv
3. Pairwise Feature Extraction & Classification -> output/matching_results.tsv
4. Automated Validation via utils/validate_submission.py
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple
import duckdb
import numpy as np

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalizer import normalize_business_name, normalize_business_address
from blocking import MultiPassBlocker
from features import extract_pairwise_features
from classifier import EntityMatcher, compute_macro_f05
import config


def preprocess_record(name: str, addr: str, country: str) -> dict:
    """Preprocess and cache normalized representations of a record."""
    return {
        "name": name,
        "addr": addr,
        "country": country,
        "n_info": normalize_business_name(name),
        "a_info": normalize_business_address(addr, country)
    }


def train_model(con: duckdb.DuckDBPyConnection) -> Tuple[EntityMatcher, float]:
    """Train the matching model on a diverse sample from training data."""
    print("\n" + "="*50)
    print("STAGE 1: TRAINING MODEL ON GROUND TRUTH")
    print("="*50)
    t0 = time.time()

    # Load 40,000 S1 records across US and India with their ground truth
    print("Loading balanced training slice from DuckDB...")
    con.execute(f"""
        CREATE OR REPLACE TABLE train_sample_s1 AS 
        (
            SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country, g.matched_entity_ids
            FROM read_csv('{config.TRAIN_DIR / "train_source1.tsv"}', delim='\\t', header=true) s1
            JOIN read_csv('{config.TRAIN_DIR / "train_ground_truth.tsv"}', delim='\\t', header=true) g 
              ON s1.entity_id = g.source1_entity_id
            WHERE s1.country = 'US'
            LIMIT {config.TRAIN_SAMPLE_PER_COUNTRY // 2}
        )
        UNION ALL
        (
            SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country, g.matched_entity_ids
            FROM read_csv('{config.TRAIN_DIR / "train_source1.tsv"}', delim='\\t', header=true) s1
            JOIN read_csv('{config.TRAIN_DIR / "train_ground_truth.tsv"}', delim='\\t', header=true) g 
              ON s1.entity_id = g.source1_entity_id
            WHERE s1.country = 'India'
            LIMIT {config.TRAIN_SAMPLE_PER_COUNTRY // 2}
        );
    """)

    con.execute("""
        CREATE OR REPLACE TABLE train_sample_gt AS
        SELECT entity_id as source1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) as matched_id
        FROM train_sample_s1
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != '';
    """)

    df_s1 = con.execute("SELECT entity_id, business_name, business_address, country FROM train_sample_s1").df()
    true_pairs = set(con.execute("SELECT source1_entity_id, matched_id FROM train_sample_gt").fetchall())
    print(f"Loaded {len(df_s1):,} training S1 entities with {len(true_pairs):,} true positive links.")

    # Load S2 and S3 corresponding to true matches + background distractors
    con.execute(f"""
        CREATE OR REPLACE TABLE train_s23 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{config.TRAIN_DIR / "train_source2.tsv"}', delim='\\t', header=true)
        WHERE entity_id IN (SELECT matched_id FROM train_sample_gt WHERE matched_id LIKE 'S2-%')
        UNION ALL
        (SELECT entity_id, business_name, business_address, country FROM read_csv('{config.TRAIN_DIR / "train_source2.tsv"}', delim='\\t', header=true) LIMIT 40000)
        UNION ALL
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{config.TRAIN_DIR / "train_source3.tsv"}', delim='\\t', header=true)
        WHERE entity_id IN (SELECT matched_id FROM train_sample_gt WHERE matched_id LIKE 'S3-%')
        UNION ALL
        (SELECT entity_id, business_name, business_address, country FROM read_csv('{config.TRAIN_DIR / "train_source3.tsv"}', delim='\\t', header=true) LIMIT 40000);
    """)

    df_s23 = con.execute("SELECT DISTINCT entity_id, business_name, business_address, country FROM train_s23").df()
    print(f"Loaded {len(df_s23):,} S2/S3 target records for training.")

    print("Preprocessing and building training index...")
    s1_cache = {row['entity_id']: preprocess_record(row['business_name'], row['business_address'], row['country']) for _, row in df_s1.iterrows()}
    s23_cache = {row['entity_id']: preprocess_record(row['business_name'], row['business_address'], row['country']) for _, row in df_s23.iterrows()}

    blocker = MultiPassBlocker(max_key_frequency=800)
    blocker.fit(s23_cache)

    print("Generating candidate training pairs...")
    candidate_pairs = []
    for s1_id, e1 in s1_cache.items():
        cands = blocker.get_candidates(e1, max_candidates=config.MAX_CANDIDATES_PER_ENTITY)
        for cid, shared_cnt in cands:
            candidate_pairs.append((s1_id, cid, shared_cnt))

    print(f"Candidate pairs generated: {len(candidate_pairs):,} (avg {len(candidate_pairs)/len(df_s1):.1f} per S1).")

    print("Extracting pairwise features...")
    X, y = [], []
    for s1_id, cid, shared_cnt in candidate_pairs:
        feats = extract_pairwise_features(s1_cache[s1_id], s23_cache[cid], shared_cnt, cid)
        label = 1 if (s1_id, cid) in true_pairs else 0
        X.append(feats)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Training feature matrix shape: {X.shape}, Positives: {np.sum(y):,} ({np.mean(y)*100:.2f}%)")

    # Split train/validation by S1 entity
    np.random.seed(42)
    s1_ids = np.array(list(s1_cache.keys()))
    np.random.shuffle(s1_ids)
    n_train = int(len(s1_ids) * 0.7)
    train_s1_set = set(s1_ids[:n_train])
    val_s1_list = list(s1_ids[n_train:])
    val_s1_set = set(val_s1_list)

    train_mask = np.array([p[0] in train_s1_set for p in candidate_pairs])
    val_mask = ~train_mask
    val_pairs = [candidate_pairs[i] for i in range(len(candidate_pairs)) if val_mask[i]]

    print("Training LightGBM model...")
    matcher = EntityMatcher(
        params=config.LGB_PARAMS,
        num_boost_round=config.NUM_BOOST_ROUND,
        threshold=config.CLASSIFICATION_THRESHOLD
    )
    matcher.train(X[train_mask], y[train_mask], X[val_mask], y[val_mask])

    # Calibrate threshold on validation set
    val_preds = matcher.predict_proba(X[val_mask])
    gt_dict = defaultdict(set)
    for s1_id, mid in true_pairs:
        gt_dict[s1_id].add(mid)

    best_thresh = config.CLASSIFICATION_THRESHOLD
    best_score = 0.0
    for thresh in np.arange(0.40, 0.75, 0.05):
        pred_dict = defaultdict(set)
        for idx, (s1_id, cid, _) in enumerate(val_pairs):
            if val_preds[idx] >= thresh:
                pred_dict[s1_id].add(cid)
        score = compute_macro_f05(gt_dict, pred_dict, val_s1_list)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"Optimal Threshold: {best_thresh:.2f} -> Validation Macro F_0.5: {best_score:.4f}")
    matcher.threshold = best_thresh

    # Save model
    model_path = config.MODEL_DIR / "lgb_matcher.txt"
    matcher.save(model_path)
    print(f"Model saved to {model_path} in {time.time() - t0:.1f}s.")
    return matcher, best_thresh


def run_country_inference(
    country: str,
    matcher: EntityMatcher,
    con: duckdb.DuckDBPyConnection,
    matching_f,
    candidate_f
):
    """
    Process one country partition end-to-end:
    Loads test data for country, blocks candidates, computes features,
    scores matches, and writes directly to open output TSV file handles.
    """
    print(f"\n--- Processing Country Partition: {country} ---")
    t0 = time.time()

    # Load test S1 for this country
    print(f"[{country}] Loading test Source 1 records...")
    df_s1 = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{config.TEST_DIR / "test_source1.tsv"}', delim='\\t', header=true)
        WHERE country = '{country}';
    """).df()

    # Load test S2 and S3 for this country
    print(f"[{country}] Loading test Source 2 and Source 3 records...")
    df_s23 = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{config.TEST_DIR / "test_source2.tsv"}', delim='\\t', header=true)
        WHERE country = '{country}'
        UNION ALL
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{config.TEST_DIR / "test_source3.tsv"}', delim='\\t', header=true)
        WHERE country = '{country}';
    """).df()

    n_s1 = len(df_s1)
    n_s23 = len(df_s23)
    print(f"[{country}] S1 Records: {n_s1:,}, Target S2/S3 Records: {n_s23:,}")

    # Build target index
    print(f"[{country}] Normalizing and indexing {n_s23:,} target records...")
    s23_cache = {}
    for _, row in df_s23.iterrows():
        s23_cache[row['entity_id']] = preprocess_record(row['business_name'], row['business_address'], row['country'])

    blocker = MultiPassBlocker(max_key_frequency=800)
    blocker.fit(s23_cache)
    print(f"[{country}] Inverted index ready with {len(blocker.filtered_index):,} keys.")

    # Process S1 entities in batches
    BATCH_SIZE = 10000
    total_matches_found = 0
    total_candidates_found = 0

    print(f"[{country}] Generating candidates and predicting matches in batches...")
    for i in range(0, n_s1, BATCH_SIZE):
        batch_df = df_s1.iloc[i : i + BATCH_SIZE]
        
        # Candidate generation for batch
        batch_pairs = []
        batch_s1_cache = {}
        for _, row in batch_df.iterrows():
            s1_id = row['entity_id']
            e1 = preprocess_record(row['business_name'], row['business_address'], row['country'])
            batch_s1_cache[s1_id] = e1
            cands = blocker.get_candidates(e1, max_candidates=config.MAX_CANDIDATES_PER_ENTITY)
            for cid, shared_cnt in cands:
                batch_pairs.append((s1_id, cid, shared_cnt))

        # Extract features for batch
        if batch_pairs:
            X_batch = []
            for s1_id, cid, shared_cnt in batch_pairs:
                feats = extract_pairwise_features(batch_s1_cache[s1_id], s23_cache[cid], shared_cnt, cid)
                X_batch.append(feats)
            X_batch = np.array(X_batch, dtype=np.float32)
            probs = matcher.predict_proba(X_batch)
        else:
            probs = np.array([])

        # Aggregate predictions per S1
        cand_map = defaultdict(list)
        match_map = defaultdict(list)
        
        for idx, (s1_id, cid, _) in enumerate(batch_pairs):
            cand_map[s1_id].append(cid)
            if probs[idx] >= matcher.threshold:
                match_map[s1_id].append(cid)

        # Write batch to output files
        for _, row in batch_df.iterrows():
            s1_id = row['entity_id']
            c_list = cand_map.get(s1_id, [])
            m_list = match_map.get(s1_id, [])
            
            cand_str = ",".join(c_list)
            match_str = ",".join(m_list)
            
            candidate_f.write(f"{s1_id}\t{cand_str}\n")
            matching_f.write(f"{s1_id}\t{match_str}\n")
            
            total_candidates_found += len(c_list)
            total_matches_found += len(m_list)

        if (i // BATCH_SIZE) % 5 == 0 or i + BATCH_SIZE >= n_s1:
            progress = min(i + BATCH_SIZE, n_s1)
            print(f"  [{country}] Processed {progress:,}/{n_s1:,} entities ({progress/n_s1*100:.1f}%)")

    elapsed = time.time() - t0
    avg_c = total_candidates_found / n_s1 if n_s1 > 0 else 0
    avg_m = total_matches_found / n_s1 if n_s1 > 0 else 0
    print(f"[{country}] Completed in {elapsed:.1f}s | Avg Candidates/S1: {avg_c:.2f} | Avg Matches/S1: {avg_m:.2f}")


def main():
    print("="*60)
    print("AMAZON ML CHALLENGE 2026: END-TO-END PIPELINE")
    print("="*60)
    total_start = time.time()

    con = duckdb.connect()

    # Step 1: Train Matching Model
    matcher, threshold = train_model(con)

    # Step 2: Test Set Inference across all countries
    print("\n" + "="*50)
    print("STAGE 2: FULL TEST SET INFERENCE")
    print("="*50)

    # Open output files with headers
    with open(config.MATCHING_OUTPUT_PATH, "w", encoding="utf-8") as matching_f, \
         open(config.CANDIDATE_OUTPUT_PATH, "w", encoding="utf-8") as candidate_f:

        matching_f.write("source1_entity_id\tmatched_entity_ids\n")
        candidate_f.write("source1_entity_id\tcandidate_entity_ids\n")

        # Process each country partition independently
        countries = ["France", "US", "India"]
        for country in countries:
            run_country_inference(country, matcher, con, matching_f, candidate_f)

    print("\n" + "="*50)
    print("STAGE 3: VERIFICATION VIA validate_submission.py")
    print("="*50)
    
    val_script = config.BASE_DIR / "student_resource" / "utils" / "validate_submission.py"
    cmd = (
        f'python "{val_script}" '
        f'--matching "{config.MATCHING_OUTPUT_PATH}" '
        f'--candidate "{config.CANDIDATE_OUTPUT_PATH}" '
        f'--test-dir "{config.TEST_DIR}"'
    )
    print(f"Running validation: {cmd}")
    ret = os.system(cmd)
    
    total_elapsed = time.time() - total_start
    print(f"\nAll operations completed in {total_elapsed/60:.2f} minutes.")
    if ret == 0:
        print("\n*** SUBMISSION VALIDATION PASSED SUCCESSFULLY! ***")
    else:
        print(f"\nValidation returned exit code: {ret}")


if __name__ == "__main__":
    main()
