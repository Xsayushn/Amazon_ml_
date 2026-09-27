import sys
import os
import duckdb
from collections import defaultdict
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address

con = duckdb.connect()

# 5,000 S1 records with ground truth
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
    (SELECT * FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true) LIMIT 50000);
""")

con.execute("""
    CREATE TABLE s3_val AS 
    SELECT * FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exploded WHERE matched_id LIKE 'S3-%')
    UNION ALL
    (SELECT * FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true) LIMIT 50000);
""")

df_s1 = con.execute("SELECT * FROM s1_val").df()
df_s2 = con.execute("SELECT DISTINCT * FROM s2_val").df()
df_s3 = con.execute("SELECT DISTINCT * FROM s3_val").df()
gt_dict = {row[0]: set(row[1].split(',')) if row[1] else set() for row in con.execute("SELECT * FROM gt_sample").fetchall()}
total_true = con.execute("SELECT count(*) FROM gt_exploded").fetchone()[0]

COMMON_WORDS = {'street', 'road', 'avenue', 'lane', 'drive', 'near', 'opposite', 'behind', 'north', 'south', 'east', 'west', 'floor', 'shop', 'block'}

def extract_keys(name, addr, country):
    keys = []
    n_info = normalize_business_name(name)
    a_info = normalize_business_address(addr, country)
    
    # 1. Name tokens
    for token in n_info['core_tokens']:
        if len(token) >= 3 and token not in COMMON_WORDS:
            keys.append(f"{country}:tok:{token}")
            
    # 2. Compressed prefix (5 and 7 chars)
    if len(n_info['compressed']) >= 5:
        keys.append(f"{country}:c5:{n_info['compressed'][:5]}")
        keys.append(f"{country}:c7:{n_info['compressed'][:7]}")
        
    # 3. Postal code
    if a_info['postal_code']:
        keys.append(f"{country}:post:{a_info['postal_code']}")
        
    # 4. Long distinctive address tokens (len >= 6)
    for w in a_info['tokens']:
        if len(w) >= 6 and not w.isdigit() and w not in COMMON_WORDS:
            keys.append(f"{country}:atok:{w}")
            
    # 5. Distinctive numbers (len >= 3)
    for d in a_info['digits']:
        if len(d) >= 3:
            keys.append(f"{country}:num:{d}")

    return set(keys)

print("Building inverted index...")
index = defaultdict(list)
for df in [df_s2, df_s3]:
    for _, row in df.iterrows():
        eid, name, addr, country = row['entity_id'], row['business_name'], row['business_address'], row['country']
        keys = extract_keys(name, addr, country)
        for k in keys:
            index[k].append(eid)

print(f"Total unique keys: {len(index):,}")

# Filter out overly common keys (stop-keys that appear in > 1000 records) to keep candidates tight
filtered_index = {k: v for k, v in index.items() if len(v) <= 800}
print(f"Keys retained after frequency filter: {len(filtered_index):,}")

print("Generating candidates and measuring recall...")
hits = 0
total_cands = 0
MAX_CANDS = 25

for _, row in df_s1.iterrows():
    s1_id, name, addr, country = row['entity_id'], row['business_name'], row['business_address'], row['country']
    keys = extract_keys(name, addr, country)
    
    cand_counts = defaultdict(int)
    for k in keys:
        if k in filtered_index:
            for cid in filtered_index[k]:
                cand_counts[cid] += 1
                
    # Sort by number of matched keys
    top_cands = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:MAX_CANDS]
    cand_set = {c[0] for c in top_cands}
    total_cands += len(cand_set)
    
    true_mids = gt_dict.get(s1_id, set())
    if true_mids:
        hits += len(true_mids & cand_set)

recall = hits / total_true
avg_cands = total_cands / len(df_s1)
print(f"\n--- Enhanced Blocking Results ---")
print(f"True positive links: {total_true:,}")
print(f"Recovered links:     {hits:,}")
print(f"Recall Ceiling:      {recall*100:.2f}%")
print(f"Avg Candidates / S1: {avg_cands:.2f}")
