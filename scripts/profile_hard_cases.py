import sys
import os
import re
import duckdb
from unidecode import unidecode
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

con = duckdb.connect()

con.execute("""
    CREATE TABLE gt AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    LIMIT 2000;
""")

con.execute("""
    CREATE TABLE gt_pairs AS
    SELECT source1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) as matched_id
    FROM gt;
""")

con.execute("""
    CREATE TABLE s1_sample AS
    SELECT s1.entity_id as s1_id, s1.business_name as s1_name, s1.business_address as s1_addr, s1.country as s1_country
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true) s1
    SEMI JOIN (SELECT DISTINCT source1_entity_id FROM gt_pairs) g ON s1.entity_id = g.source1_entity_id;
""")

con.execute("""
    CREATE TABLE s23_sample AS
    SELECT entity_id as m_id, business_name as m_name, business_address as m_addr, country as m_country
    FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true)
    SEMI JOIN (SELECT DISTINCT matched_id FROM gt_pairs WHERE matched_id LIKE 'S2-%') g ON entity_id = g.matched_id
    UNION ALL
    SELECT entity_id as m_id, business_name as m_name, business_address as m_addr, country as m_country
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    SEMI JOIN (SELECT DISTINCT matched_id FROM gt_pairs WHERE matched_id LIKE 'S3-%') g ON entity_id = g.matched_id;
""")

con.execute("""
    CREATE TABLE joined_pairs AS
    SELECT p.source1_entity_id, p.matched_id,
           s1.s1_name, s1.s1_addr, s1.s1_country,
           s23.m_name, s23.m_addr, s23.m_country
    FROM gt_pairs p
    JOIN s1_sample s1 ON p.source1_entity_id = s1.s1_id
    JOIN s23_sample s23 ON p.matched_id = s23.m_id;
""")

df_pairs = con.execute("SELECT * FROM joined_pairs").df()

def clean_str(s):
    if not s or s is None: return ""
    s = unidecode(str(s).lower())
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return " ".join(s.split())

count = 0
print("--- Difficult Match Cases (name_sim < 80 AND addr_sim < 80) ---")
for _, row in df_pairs.iterrows():
    c1_n = clean_str(row['s1_name'])
    c2_n = clean_str(row['m_name'])
    c1_a = clean_str(row['s1_addr'])
    c2_a = clean_str(row['m_addr'])
    
    sim_name = fuzz.token_sort_ratio(c1_n, c2_n) if (c1_n and c2_n) else 0
    sim_addr = fuzz.token_sort_ratio(c1_a, c2_a) if (c1_a and c2_a) else 0
    
    if sim_name < 70 and sim_addr < 70:
        count += 1
        print(f"\n[Case {count}] Country: {row['s1_country']}")
        print(f"  S1:   Name: '{row['s1_name']}' | Addr: '{row['s1_addr']}'")
        print(f"  S2/3: Name: '{row['m_name']}' | Addr: '{row['m_addr']}'")
        print(f"  Sims: Name token_sort={sim_name}, Addr token_sort={sim_addr}")
        if count >= 8:
            break
