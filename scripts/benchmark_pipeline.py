import sys
import os
import duckdb
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address

print("=== Starting Full Pipeline Benchmark on Validation Slice ===")

con = duckdb.connect()

# 5,000 S1 entities
con.execute("""
    CREATE TABLE gt_sample AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    LIMIT 5000;
""")

con.execute("""
    CREATE TABLE gt_exploded AS
    SELECT source1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) as matched_id
    FROM gt_sample
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != '';
""")

con.execute("""
    CREATE TABLE s1_val AS 
    SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true) s1
    JOIN gt_sample g ON s1.entity_id = g.source1_entity_id;
""")

con.execute("""
    CREATE TABLE s2_val AS 
    SELECT * FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exploded WHERE matched_id LIKE 'S2-%')
    UNION ALL
    (SELECT * FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true) LIMIT 40000);
""")

con.execute("""
    CREATE TABLE s3_val AS 
    SELECT * FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exploded WHERE matched_id LIKE 'S3-%')
    UNION ALL
    (SELECT * FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true) LIMIT 40000);
""")

df_s1 = con.execute("SELECT * FROM s1_val").df()
df_s2 = con.execute("SELECT DISTINCT * FROM s2_val").df()
df_s3 = con.execute("SELECT DISTINCT * FROM s3_val").df()

all_s1_ids = df_s1['entity_id'].tolist()
gt_dict = {row[0]: set(row[1].split(',')) if row[1] else set() for row in con.execute("SELECT * FROM gt_sample").fetchall()}
true_pairs_set = set(con.execute("SELECT source1_entity_id, matched_id FROM gt_exploded").fetchall())

print(f"Entities: S1={len(df_s1)}, True Pairs={len(true_pairs_set)}")

# Preprocess records
print("Normalizing records...")
s1_cache = {}
for _, row in df_s1.iterrows():
    s1_cache[row['entity_id']] = {
        'name': row['business_name'],
        'addr': row['business_address'],
        'country': row['country'],
        'n_info': normalize_business_name(row['business_name']),
        'a_info': normalize_business_address(row['business_address'], row['country'])
    }

s23_cache = {}
for df in [df_s2, df_s3]:
    for _, row in df.iterrows():
        s23_cache[row['entity_id']] = {
            'name': row['business_name'],
            'addr': row['business_address'],
            'country': row['country'],
            'n_info': normalize_business_name(row['business_name']),
            'a_info': normalize_business_address(row['business_address'], row['country'])
        }

# Blocking index
print("Building blocking index...")
COMMON_WORDS = {'street', 'road', 'avenue', 'lane', 'drive', 'near', 'opposite', 'behind', 'north', 'south', 'east', 'west', 'floor', 'shop', 'block'}

def extract_keys(entry):
    keys = []
    country = entry['country']
    n_info = entry['n_info']
    a_info = entry['a_info']
    
    for token in n_info['core_tokens']:
        if len(token) >= 3 and token not in COMMON_WORDS:
            keys.append(f"{country}:tok:{token}")
            
    if len(n_info['compressed']) >= 5:
        keys.append(f"{country}:c5:{n_info['compressed'][:5]}")
        keys.append(f"{country}:c7:{n_info['compressed'][:7]}")
        
    if a_info['postal_code']:
        keys.append(f"{country}:post:{a_info['postal_code']}")
        
    for w in a_info['tokens']:
        if len(w) >= 6 and not w.isdigit() and w not in COMMON_WORDS:
            keys.append(f"{country}:atok:{w}")
            
    for d in a_info['digits']:
        if len(d) >= 3:
            keys.append(f"{country}:num:{d}")

    return set(keys)

index = defaultdict(list)
for eid, entry in s23_cache.items():
    for k in extract_keys(entry):
        index[k].append(eid)

filtered_index = {k: v for k, v in index.items() if len(v) <= 800}

# Candidate Generation
print("Generating candidate pairs...")
candidate_pairs = []
cand_dict = {}

for s1_id in all_s1_ids:
    entry1 = s1_cache[s1_id]
    cand_counts = defaultdict(int)
    for k in extract_keys(entry1):
        if k in filtered_index:
            for cid in filtered_index[k]:
                cand_counts[cid] += 1
                
    top_cands = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    c_list = [c[0] for c in top_cands]
    cand_dict[s1_id] = set(c_list)
    for cid, cnt in top_cands:
        candidate_pairs.append((s1_id, cid, cnt))

print(f"Total candidate pairs: {len(candidate_pairs):,} (avg {len(candidate_pairs)/len(all_s1_ids):.1f} per S1)")

# Feature extraction function
def compute_features(s1_id, cid, shared_keys):
    e1 = s1_cache[s1_id]
    e2 = s23_cache[cid]
    
    n1, n2 = e1['n_info'], e2['n_info']
    a1, a2 = e1['a_info'], e2['a_info']
    
    c1_clean, c2_clean = n1['clean'], n2['clean']
    c1_core, c2_core = n1['core'], n2['core']
    
    name_sort = fuzz.token_sort_ratio(c1_clean, c2_clean) if c1_clean and c2_clean else 0.0
    name_set = fuzz.token_set_ratio(c1_clean, c2_clean) if c1_clean and c2_clean else 0.0
    name_core_sort = fuzz.token_sort_ratio(c1_core, c2_core) if c1_core and c2_core else 0.0
    name_ratio = fuzz.ratio(c1_clean, c2_clean) if c1_clean and c2_clean else 0.0
    
    comp_exact = 1.0 if (n1['compressed'] and n1['compressed'] == n2['compressed']) else 0.0
    comp_ratio = fuzz.ratio(n1['compressed'], n2['compressed']) if n1['compressed'] and n2['compressed'] else 0.0
    
    # Name tokens overlap
    t1_n, t2_n = set(n1['tokens']), set(n2['tokens'])
    name_jaccard = len(t1_n & t2_n) / len(t1_n | t2_n) if (t1_n | t2_n) else 0.0
    name_overlap = len(t1_n & t2_n) / min(len(t1_n), len(t2_n)) if min(len(t1_n), len(t2_n)) > 0 else 0.0
    
    # Address features
    addr1, addr2 = a1['clean'], a2['clean']
    addr_sort = fuzz.token_sort_ratio(addr1, addr2) if addr1 and addr2 else 0.0
    addr_set = fuzz.token_set_ratio(addr1, addr2) if addr1 and addr2 else 0.0
    
    t1_a, t2_a = a1.get('token_set', set()), a2.get('token_set', set())
    addr_jaccard = len(t1_a & t2_a) / len(t1_a | t2_a) if (t1_a | t2_a) else 0.0
    addr_overlap = len(t1_a & t2_a) / min(len(t1_a), len(t2_a)) if min(len(t1_a), len(t2_a)) > 0 else 0.0
    
    # Postal & digits
    postal_match = 1.0 if (a1['postal_code'] and a1['postal_code'] == a2['postal_code']) else (0.0 if (a1['postal_code'] and a2['postal_code']) else -1.0)
    
    d1, d2 = a1['digits'], a2['digits']
    digit_overlap = len(d1 & d2) / len(d1 | d2) if (d1 | d2) else -1.0
    
    addr_empty = 1.0 if not addr2 else 0.0
    src_type = 1.0 if cid.startswith('S2-') else 2.0
    
    max_sim = max(name_sort, addr_sort)
    min_sim = min(name_sort, addr_sort)
    mean_sim = (name_sort + addr_sort) / 2.0
    
    return [
        name_sort, name_set, name_core_sort, name_ratio, comp_exact, comp_ratio,
        name_jaccard, name_overlap,
        addr_sort, addr_set, addr_jaccard, addr_overlap, postal_match, digit_overlap,
        addr_empty, src_type, shared_keys, max_sim, min_sim, mean_sim
    ]

print("Extracting features for candidate pairs...")
X = []
y = []
for s1_id, cid, shared_cnt in candidate_pairs:
    feats = compute_features(s1_id, cid, shared_cnt)
    label = 1 if (s1_id, cid) in true_pairs_set else 0
    X.append(feats)
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"Feature matrix shape: {X.shape}, Positive rate: {np.mean(y)*100:.2f}%")

# Train/Val Split (by S1 entity to prevent data leakage!)
np.random.seed(42)
unique_s1 = np.array(all_s1_ids)
np.random.shuffle(unique_s1)
n_train_s1 = int(len(unique_s1) * 0.7)
train_s1_set = set(unique_s1[:n_train_s1])
val_s1_set = set(unique_s1[n_train_s1:])

train_mask = np.array([p[0] in train_s1_set for p in candidate_pairs])
val_mask = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
val_pairs = [candidate_pairs[i] for i in range(len(candidate_pairs)) if val_mask[i]]

print(f"Train pairs: {len(X_train):,}, Val pairs: {len(X_val):,}")

# Train LightGBM model
print("Training LightGBM model...")
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.08,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.85,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[val_data]
)

# Predict probabilities on validation set
val_preds = model.predict(X_val)

# Optimize threshold specifically for macro F0.5 on validation entities
def compute_macro_f05(gt_dict, pred_dict, s1_id_list):
    scores = []
    for s1_id in s1_id_list:
        y_true = gt_dict.get(s1_id, set())
        y_pred = pred_dict.get(s1_id, set())
        
        if not y_true:
            scores.append(1.0 if not y_pred else 0.0)
            continue
            
        if not y_pred:
            scores.append(0.0)
            continue
            
        tp = len(y_true & y_pred)
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if prec + rec == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            scores.append(f05)
            
    return float(np.mean(scores))

best_thresh = 0.5
best_score = 0.0
val_s1_list = list(val_s1_set)

print("\n--- Optimizing Threshold for Macro F_0.5 ---")
for thresh in np.arange(0.35, 0.85, 0.05):
    pred_dict = defaultdict(set)
    for idx, (s1_id, cid, _) in enumerate(val_pairs):
        if val_preds[idx] >= thresh:
            pred_dict[s1_id].add(cid)
            
    score = compute_macro_f05(gt_dict, pred_dict, val_s1_list)
    print(f"Threshold: {thresh:.2f} -> Validation Macro F_0.5: {score:.4f}")
    if score > best_score:
        best_score = score
        best_thresh = thresh

print(f"\n==========================================")
print(f"BEST THRESHOLD: {best_thresh:.2f}")
print(f"BEST VALIDATION MACRO F_0.5: {best_score:.4f}")
print(f"==========================================")
