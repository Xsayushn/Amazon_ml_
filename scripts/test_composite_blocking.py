import sys
import duckdb
from collections import defaultdict
import time

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address

con = duckdb.connect()

# 2,000 S1 records with ground truth
con.execute("""
    CREATE TABLE gt_sample AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    LIMIT 2000;
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
    (SELECT entity_id, business_name, business_address, country FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true) LIMIT 60000)
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exp WHERE matched_id LIKE 'S3-%')
    UNION ALL
    (SELECT entity_id, business_name, business_address, country FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true) LIMIT 60000);
""")

df_s1 = con.execute("SELECT * FROM s1_test").df()
df_s23 = con.execute("SELECT DISTINCT * FROM s23_test").df()
gt_dict = {r[0]: set(r[1].split(',')) if r[1] else set() for r in con.execute("SELECT * FROM gt_sample").fetchall()}
total_true = con.execute("SELECT count(*) FROM gt_exp").fetchone()[0]

STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'near', 'opposite', 'behind', 
    'floor', 'shop', 'block', 'unit', 'apartment', 'suite', 'road', 
    'street', 'avenue', 'lane', 'drive', 'highway', 'parkway', 'place', 'city', 'state'
}

def extract_composite_keys(entry):
    keys = []
    country = entry['country']
    n_info = entry['n_info']
    a_info = entry['a_info']
    
    name_tokens = [t for t in n_info['core_tokens'] if len(t) >= 2 and t not in STOPWORDS]
    addr_tokens = [t for t in a_info['tokens'] if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()]
    digits = list(a_info['digits'])
    postal = a_info['postal_code']
    comp = n_info['compressed']
    
    # 1. Name Bigram (two core words together)
    if len(name_tokens) >= 2:
        keys.append(f"{country}:n2:{name_tokens[0]}_{name_tokens[1]}")
    elif len(name_tokens) == 1:
        keys.append(f"{country}:n1:{name_tokens[0]}")
        
    # 2. Compressed name prefix (first 7 chars)
    if len(comp) >= 6:
        keys.append(f"{country}:comp7:{comp[:7]}")
        
    # 3. Composite Key: Street Number + Street Token
    if digits and addr_tokens:
        for d in digits[:2]:
            for at in addr_tokens[:3]:
                keys.append(f"{country}:num_addr:{d}_{at}")
                
    # 4. Composite Key: Street Number + First Name Token
    if digits and name_tokens:
        for d in digits[:2]:
            keys.append(f"{country}:num_name:{d}_{name_tokens[0]}")
            
    # 5. Composite Key: Postal Code + Name Token
    if postal and name_tokens:
        keys.append(f"{country}:post_name:{postal}_{name_tokens[0]}")
        
    # 6. Composite Key: Name Token + City/Address Token
    if name_tokens and addr_tokens:
        keys.append(f"{country}:name_addr:{name_tokens[0]}_{addr_tokens[-1]}")
        if len(addr_tokens) >= 2:
            keys.append(f"{country}:name_addr:{name_tokens[0]}_{addr_tokens[-2]}")
            
    return set(keys)

print("Precomputing records...")
s1_cache = {r['entity_id']: {'country': r['country'], 'n_info': normalize_business_name(r['business_name']), 'a_info': normalize_business_address(r['business_address'], r['country'])} for _, r in df_s1.iterrows()}
s23_cache = {r['entity_id']: {'country': r['country'], 'n_info': normalize_business_name(r['business_name']), 'a_info': normalize_business_address(r['business_address'], r['country'])} for _, r in df_s23.iterrows()}

index = defaultdict(list)
for eid, entry in s23_cache.items():
    for k in extract_composite_keys(entry):
        index[k].append(eid)

print(f"Total composite keys: {len(index):,}")

# Measure recall
hits = 0
total_cands = 0
for s1_id, e1 in s1_cache.items():
    cand_counts = defaultdict(int)
    for k in extract_composite_keys(e1):
        if k in index:
            for cid in index[k]:
                cand_counts[cid] += 1
                
    top_cands = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    c_set = {c[0] for c in top_cands}
    total_cands += len(c_set)
    
    true_mids = gt_dict.get(s1_id, set())
    if true_mids:
        hits += len(true_mids & c_set)

recall = hits / total_true
avg_cands = total_cands / len(df_s1)
print(f"\n--- COMPOSITE BLOCKING EVALUATION ---")
print(f"True positive links: {total_true:,}")
print(f"Recovered links:     {hits:,}")
print(f"Candidate Recall:    {recall*100:.2f}%")
print(f"Avg Candidates / S1: {avg_cands:.2f}")
