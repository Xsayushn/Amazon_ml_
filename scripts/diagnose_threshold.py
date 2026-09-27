import sys
import os
import duckdb
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address
from features import extract_pairwise_features
from classifier import compute_macro_f05
from blocking import MultiPassBlocker

print("Loading saved model from models/lgb_matcher.txt...")
model = lgb.Booster(model_file="models/lgb_matcher.txt")

con = duckdb.connect()

# Load 2,000 S1 records with ground truth
con.execute("""
    CREATE TABLE gt_sample AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    LIMIT 2000;
""")

con.execute("""
    CREATE TABLE s1_test AS 
    SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true) s1
    JOIN gt_sample g ON s1.entity_id = g.source1_entity_id;
""")

con.execute("""
    CREATE TABLE gt_exp AS
    SELECT source1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) as matched_id
    FROM gt_sample
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != '';
""")

con.execute("""
    CREATE TABLE s23_test AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exp WHERE matched_id LIKE 'S2-%')
    UNION ALL
    (SELECT entity_id, business_name, business_address, country FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true) LIMIT 50000)
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exp WHERE matched_id LIKE 'S3-%')
    UNION ALL
    (SELECT entity_id, business_name, business_address, country FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true) LIMIT 50000);
""")

df_s1 = con.execute("SELECT * FROM s1_test").df()
df_s23 = con.execute("SELECT DISTINCT * FROM s23_test").df()
gt_dict = {r[0]: set(r[1].split(',')) if r[1] else set() for r in con.execute("SELECT * FROM gt_sample").fetchall()}
all_s1 = df_s1['entity_id'].tolist()

print(f"Loaded: S1={len(df_s1)}, S23={len(df_s23)}")

s1_cache = {r['entity_id']: {'name': r['business_name'], 'addr': r['business_address'], 'country': r['country'], 'n_info': normalize_business_name(r['business_name']), 'a_info': normalize_business_address(r['business_address'], r['country'])} for _, r in df_s1.iterrows()}
s23_cache = {r['entity_id']: {'name': r['business_name'], 'addr': r['business_address'], 'country': r['country'], 'n_info': normalize_business_name(r['business_name']), 'a_info': normalize_business_address(r['business_address'], r['country'])} for _, r in df_s23.iterrows()}

blocker = MultiPassBlocker(max_key_frequency=800)
blocker.fit(s23_cache)

pairs = []
for s1_id, e1 in s1_cache.items():
    cands = blocker.get_candidates(e1, max_candidates=15)
    for cid, cnt in cands:
        pairs.append((s1_id, cid, cnt))

print(f"Candidate pairs: {len(pairs):,}")

X = []
for s1_id, cid, cnt in pairs:
    X.append(extract_pairwise_features(s1_cache[s1_id], s23_cache[cid], cnt, cid))
X = np.array(X, dtype=np.float32)

probs = model.predict(X)

print("\n--- Macro F_0.5 at different thresholds with CURRENT model ---")
for t in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    pdict = defaultdict(set)
    for i, (s1_id, cid, _) in enumerate(pairs):
        if probs[i] >= t:
            pdict[s1_id].add(cid)
    f05 = compute_macro_f05(gt_dict, pdict, all_s1)
    
    # Also calculate macro precision and recall
    tot_p, tot_r = [], []
    for s1_id in all_s1:
        y_true = gt_dict.get(s1_id, set())
        y_pred = pdict.get(s1_id, set())
        if y_true:
            tp = len(y_true & y_pred)
            tot_p.append(tp / len(y_pred) if y_pred else 0)
            tot_r.append(tp / len(y_true))
    print(f"Threshold {t:.2f} -> Macro F_0.5: {f05:.4f} | Prec: {np.mean(tot_p):.4f} | Rec: {np.mean(tot_r):.4f} | Predicted Empty: {sum(1 for s in all_s1 if not pdict.get(s))}")
