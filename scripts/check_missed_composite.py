import sys
import duckdb
from collections import defaultdict

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address

con = duckdb.connect()

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
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exp WHERE matched_id LIKE 'S3-%');
""")

df_s1 = con.execute("SELECT * FROM s1_test").df().set_index('entity_id')
df_s23 = con.execute("SELECT * FROM s23_test").df().set_index('entity_id')
gt_pairs = con.execute("SELECT * FROM gt_exp").fetchall()

def get_keys(name, addr, country):
    n_info = normalize_business_name(name)
    a_info = normalize_business_address(addr, country)
    core = n_info['core_tokens']
    digits = sorted(list(a_info['digits']))
    addr_tokens = [t for t in a_info['tokens'] if not t.isdigit() and len(t) >= 3]
    comp = n_info['compressed']
    postal = a_info['postal_code']
    
    keys = []
    if n_info['core'] and len(n_info['core']) >= 4:
        keys.append(f"{country}:c_name:{n_info['core']}")
    if len(core) >= 2:
        keys.append(f"{country}:n2:{core[0]}_{core[1]}")
    elif len(core) == 1 and len(core[0]) >= 3:
        keys.append(f"{country}:n1:{core[0]}")
    if len(comp) >= 6:
        keys.append(f"{country}:cp6:{comp[:6]}")
    if len(comp) >= 8:
        keys.append(f"{country}:cp8:{comp[:8]}")
    if digits and addr_tokens:
        for d in digits[:3]:
            for at in addr_tokens[:4]:
                keys.append(f"{country}:d_at:{d}_{at}")
    if digits and core:
        for d in digits[:2]:
            keys.append(f"{country}:d_nt:{d}_{core[0]}")
    if postal and core:
        keys.append(f"{country}:post_nt:{postal}_{core[0]}")
    if postal and digits:
        for d in digits[:2]:
            keys.append(f"{country}:post_d:{postal}_{d}")
    if addr_tokens and core:
        keys.append(f"{country}:city_nt:{addr_tokens[-1]}_{core[0]}")
    return set(keys)

missed = 0
for s1_id, m_id in gt_pairs:
    if s1_id not in df_s1.index or m_id not in df_s23.index:
        continue
    r1 = df_s1.loc[s1_id]
    r2 = df_s23.loc[m_id]
    k1 = get_keys(r1['business_name'], r1['business_address'], r1['country'])
    k2 = get_keys(r2['business_name'], r2['business_address'], r2['country'])
    
    if not (k1 & k2):
        missed += 1
        if missed <= 8:
            print(f"\n[Missed {missed}] {r1['country']}")
            print(f"  S1:   '{r1['business_name']}' | '{r1['business_address']}'")
            print(f"  S23:  '{r2['business_name']}' | '{r2['business_address']}'")

print(f"\nTotal true pairs: {len(gt_pairs)}, Missed: {missed} ({missed/len(gt_pairs)*100:.2f}%)")
