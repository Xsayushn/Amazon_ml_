import sys
import os
import duckdb
from collections import defaultdict
import time

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address

t0 = time.time()
print("Loading validation sample from DuckDB...")
con = duckdb.connect()

# Load 5000 S1 records with their ground truth
con.execute("""
    CREATE TABLE gt_sample AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    LIMIT 5000;
""")

con.execute("""
    CREATE TABLE s1_val AS 
    SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true) s1
    JOIN gt_sample g ON s1.entity_id = g.source1_entity_id;
""")

# Load S2 and S3 for India and US corresponding to the validation set
# We can load 50,000 records from S2 and S3 to test candidate retrieval speed and recall
# Also ensure all true matches are loaded so recall can be measured accurately
con.execute("""
    CREATE TABLE gt_exploded AS
    SELECT source1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) as matched_id
    FROM gt_sample
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != '';
""")

total_true_matches = con.execute("SELECT count(*) FROM gt_exploded").fetchone()[0]
print(f"Validation set: 5,000 S1 entities, {total_true_matches:,} true positive links.")

# Load all true matching targets + 50k background distractors
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
gt_dict = {row[0]: set(row[1].split(',')) if row[1] else set() for row in con.execute("SELECT * FROM gt_sample").fetchall()}

print(f"Loaded: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)} in {time.time()-t0:.1f}s")

# Build blocking index
print("Building multi-pass blocking index...")
t1 = time.time()

def extract_keys(name, addr, country):
    keys = []
    n_info = normalize_business_name(name)
    a_info = normalize_business_address(addr, country)
    
    # Key 1: country + first 2 core tokens
    if n_info['core_tokens']:
        prefix2 = "_".join(n_info['core_tokens'][:2])
        keys.append(f"{country}:name2:{prefix2}")
        # Key 2: country + first core token
        keys.append(f"{country}:name1:{n_info['core_tokens'][0]}")
    
    # Key 3: country + compressed prefix (6 chars)
    if len(n_info['compressed']) >= 5:
        keys.append(f"{country}:comp:{n_info['compressed'][:6]}")
        
    # Key 4: country + postal code
    if a_info['postal_code']:
        keys.append(f"{country}:post:{a_info['postal_code']}")
        
    # Key 5: country + street digit + first non-trivial addr token
    addr_words = [w for w in a_info['tokens'] if len(w) > 3 and not w.isdigit()]
    if a_info['digits'] and addr_words:
        first_digit = sorted(list(a_info['digits']))[0]
        keys.append(f"{country}:addr_num:{first_digit}_{addr_words[0]}")
        
    return keys, n_info, a_info

# Index S2 and S3
index = defaultdict(list)
s23_records = {}

for df in [df_s2, df_s3]:
    for _, row in df.iterrows():
        eid, name, addr, country = row['entity_id'], row['business_name'], row['business_address'], row['country']
        keys, n_info, a_info = extract_keys(name, addr, country)
        s23_records[eid] = (name, addr, country, n_info, a_info)
        for k in set(keys):
            index[k].append(eid)

print(f"Index built with {len(index):,} unique keys in {time.time()-t1:.1f}s")

# Candidate generation
print("Generating candidate pairs for S1...")
t2 = time.time()
s1_candidates = {}
hits = 0
total_candidates = 0

for _, row in df_s1.iterrows():
    s1_id, name, addr, country = row['entity_id'], row['business_name'], row['business_address'], row['country']
    keys, n_info, a_info = extract_keys(name, addr, country)
    
    # Pool candidates and count key matches
    cand_counts = defaultdict(int)
    for k in set(keys):
        if k in index:
            for cid in index[k]:
                cand_counts[cid] += 1
                
    # Sort candidates by match strength and keep top 20
    sorted_cands = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    cand_ids = [c[0] for c in sorted_cands]
    s1_candidates[s1_id] = set(cand_ids)
    total_candidates += len(cand_ids)
    
    # Check ground truth recall
    true_mids = gt_dict.get(s1_id, set())
    if true_mids:
        found_mids = true_mids & s1_candidates[s1_id]
        hits += len(found_mids)

recall = hits / total_true_matches if total_true_matches > 0 else 0
avg_cands = total_candidates / len(df_s1)

print(f"\n--- Candidate Generation Performance ---")
print(f"Total True Positive Links: {total_true_matches:,}")
print(f"Recovered in Candidate Set: {hits:,}")
print(f"Candidate Recall Ceiling:  {recall*100:.2f}%")
print(f"Average Candidates per S1: {avg_cands:.2f}")
print(f"Total Candidate Generation Time: {time.time()-t2:.2f}s")
