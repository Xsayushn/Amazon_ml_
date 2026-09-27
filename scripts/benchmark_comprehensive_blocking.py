import sys
import duckdb
from collections import defaultdict
import time
import re

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address

con = duckdb.connect()

print("Loading 10,000 ground truth training entities...")
con.execute("""
    CREATE TABLE gt_sample AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    LIMIT 10000;
""")

con.execute("""
    CREATE TABLE gt_exp AS
    SELECT source1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) as matched_id
    FROM gt_sample
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != '';
""")

con.execute("""
    CREATE TABLE s1_test AS 
    SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true) s1
    JOIN gt_sample g ON s1.entity_id = g.source1_entity_id;
""")

con.execute("""
    CREATE TABLE s23_test AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exp WHERE matched_id LIKE 'S2-%')
    UNION ALL
    (SELECT entity_id, business_name, business_address, country FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true) LIMIT 100000)
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exp WHERE matched_id LIKE 'S3-%')
    UNION ALL
    (SELECT entity_id, business_name, business_address, country FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true) LIMIT 100000);
""")

df_s1 = con.execute("SELECT * FROM s1_test").df()
df_s23 = con.execute("SELECT DISTINCT * FROM s23_test").df()
gt_dict = {r[0]: set(r[1].split(',')) if r[1] else set() for r in con.execute("SELECT * FROM gt_sample").fetchall()}
total_true = con.execute("SELECT count(*) FROM gt_exp").fetchone()[0]

print(f"Loaded: S1={len(df_s1):,}, S2/3 targets={len(df_s23):,}, True links={total_true:,}")

# Let's test what keys give maximum recall
def get_all_possible_keys(name, addr, country):
    keys = []
    n_info = normalize_business_name(name)
    a_info = normalize_business_address(addr, country)
    
    # Clean core name
    core_tokens = n_info['core_tokens']
    addr_tokens = [t for t in a_info['tokens'] if not t.isdigit() and len(t) >= 3]
    digits = sorted(list(a_info['digits']))
    comp = n_info['compressed']
    postal = a_info['postal_code']
    
    # 1. Exact core name (if length >= 4)
    if n_info['core'] and len(n_info['core']) >= 4:
        keys.append(f"{country}:c_name:{n_info['core']}")
        
    # 2. First 2 core name tokens
    if len(core_tokens) >= 2:
        keys.append(f"{country}:n2:{core_tokens[0]}_{core_tokens[1]}")
    elif len(core_tokens) == 1 and len(core_tokens[0]) >= 3:
        keys.append(f"{country}:n1:{core_tokens[0]}")
        
    # 3. Compressed prefix (6 and 8 chars)
    if len(comp) >= 6:
        keys.append(f"{country}:cp6:{comp[:6]}")
    if len(comp) >= 8:
        keys.append(f"{country}:cp8:{comp[:8]}")
        
    # 4. Street number + Street token (KEY INSIGHT: any digit + any addr word)
    if digits and addr_tokens:
        for d in digits[:3]:
            for at in addr_tokens[:4]:
                keys.append(f"{country}:d_at:{d}_{at}")
                
    # 5. Street number + Name token
    if digits and core_tokens:
        for d in digits[:2]:
            keys.append(f"{country}:d_nt:{d}_{core_tokens[0]}")
            
    # 6. Postal code + Name token
    if postal and core_tokens:
        keys.append(f"{country}:post_nt:{postal}_{core_tokens[0]}")
        
    # 7. Postal code + Street number
    if postal and digits:
        for d in digits[:2]:
            keys.append(f"{country}:post_d:{postal}_{d}")
            
    # 8. City (last token) + First name token
    if addr_tokens and core_tokens:
        keys.append(f"{country}:city_nt:{addr_tokens[-1]}_{core_tokens[0]}")

    return set(keys)

print("Building index with comprehensive composite keys...")
index = defaultdict(list)
for _, r in df_s23.iterrows():
    eid = r['entity_id']
    for k in get_all_possible_keys(r['business_name'], r['business_address'], r['country']):
        index[k].append(eid)

print(f"Total keys in index: {len(index):,}")

# Filter out only truly massive keys (appearing in > 2000 records)
filtered_index = {k: v for k, v in index.items() if len(v) <= 1500}
print(f"Retained keys: {len(filtered_index):,}")

# Test recall with candidate ranking
hits = 0
total_cands = 0
MAX_C = 20

for _, r in df_s1.iterrows():
    s1_id = r['entity_id']
    keys = get_all_possible_keys(r['business_name'], r['business_address'], r['country'])
    
    cand_weights = defaultdict(float)
    for k in keys:
        if k in filtered_index:
            bucket = filtered_index[k]
            # IDF weighting: rarer keys carry more weight!
            weight = 1.0 / (len(bucket) + 1.0)
            for cid in bucket:
                cand_weights[cid] += weight
                
    top_cands = sorted(cand_weights.items(), key=lambda x: x[1], reverse=True)[:MAX_C]
    c_set = {c[0] for c in top_cands}
    total_cands += len(c_set)
    
    true_mids = gt_dict.get(s1_id, set())
    if true_mids:
        hits += len(true_mids & c_set)

print(f"\n==========================================")
print(f"Total True Positive Links: {total_true:,}")
print(f"Recovered Links:          {hits:,}")
print(f"CANDIDATE RECALL CEILING: {hits/total_true*100:.2f}%")
print(f"Avg Candidates / S1:      {total_cands/len(df_s1):.2f}")
print(f"==========================================")
