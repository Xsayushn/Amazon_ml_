import sys
import os
import duckdb
from collections import defaultdict

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address

con = duckdb.connect()

con.execute("""
    CREATE TABLE gt_sample AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    LIMIT 1000;
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
    CREATE TABLE s23_val AS 
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exploded WHERE matched_id LIKE 'S2-%')
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM gt_exploded WHERE matched_id LIKE 'S3-%');
""")

df_s1 = con.execute("SELECT * FROM s1_val").df().set_index('entity_id')
df_s23 = con.execute("SELECT * FROM s23_val").df().set_index('entity_id')
gt_pairs = con.execute("SELECT * FROM gt_exploded").fetchall()

def extract_keys(name, addr, country):
    keys = []
    n_info = normalize_business_name(name)
    a_info = normalize_business_address(addr, country)
    
    if n_info['core_tokens']:
        keys.append(f"{country}:name2:{'_'.join(n_info['core_tokens'][:2])}")
        keys.append(f"{country}:name1:{n_info['core_tokens'][0]}")
    
    if len(n_info['compressed']) >= 5:
        keys.append(f"{country}:comp:{n_info['compressed'][:6]}")
        
    if a_info['postal_code']:
        keys.append(f"{country}:post:{a_info['postal_code']}")
        
    addr_words = [w for w in a_info['tokens'] if len(w) > 3 and not w.isdigit()]
    if a_info['digits'] and addr_words:
        first_digit = sorted(list(a_info['digits']))[0]
        keys.append(f"{country}:addr_num:{first_digit}_{addr_words[0]}")
        
    return set(keys), n_info, a_info

missed = 0
print("--- Analyzing Missed Matches in Current 5-Key Blocking ---")
for s1_id, m_id in gt_pairs:
    if s1_id not in df_s1.index or m_id not in df_s23.index:
        continue
    r1 = df_s1.loc[s1_id]
    r2 = df_s23.loc[m_id]
    
    k1, n1, a1 = extract_keys(r1['business_name'], r1['business_address'], r1['country'])
    k2, n2, a2 = extract_keys(r2['business_name'], r2['business_address'], r2['country'])
    
    shared = k1 & k2
    if not shared:
        missed += 1
        if missed <= 10:
            print(f"\n[Missed Pair {missed}] Country: {r1['country']}")
            print(f"  S1:   Name: '{r1['business_name']}' | Addr: '{r1['business_address']}'")
            print(f"  S2/3: Name: '{r2['business_name']}' | Addr: '{r2['business_address']}'")
            print(f"  Tokens S1: Name={n1['core_tokens']}, Addr={a1['tokens'][:6]}")
            print(f"  Tokens S2: Name={n2['core_tokens']}, Addr={a2['tokens'][:6]}")
print(f"\nTotal analyzed: {len(gt_pairs)}, Missed: {missed} ({missed/len(gt_pairs)*100:.2f}%)")
