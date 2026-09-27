import sys
import os
import re
from collections import Counter
import duckdb
from unidecode import unidecode
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

print("Connecting to DuckDB in-memory database...")
con = duckdb.connect()

# Read 50k rows from train ground truth
con.execute("""
    CREATE TABLE gt AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    LIMIT 20000;
""")

print("Ground truth sample loaded.")
# Explode the matched_entity_ids
con.execute("""
    CREATE TABLE gt_pairs AS
    SELECT source1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) as matched_id
    FROM gt;
""")

print("Total pairs in sample:", con.execute("SELECT count(*) FROM gt_pairs").fetchone()[0])

# Join with train_source1
con.execute("""
    CREATE TABLE s1_sample AS
    SELECT s1.entity_id as s1_id, s1.business_name as s1_name, s1.business_address as s1_addr, s1.country as s1_country
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true) s1
    SEMI JOIN (SELECT DISTINCT source1_entity_id FROM gt_pairs) g ON s1.entity_id = g.source1_entity_id;
""")

print("S1 sample loaded:", con.execute("SELECT count(*) FROM s1_sample").fetchone()[0])

# Load S2 and S3 matching records
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

print("S23 sample loaded:", con.execute("SELECT count(*) FROM s23_sample").fetchone()[0])

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
print(f"Loaded {len(df_pairs)} joined true positive pairs!")

# Now analyze matching signals
exact_name = 0
exact_addr = 0
norm_name_exact = 0
norm_addr_exact = 0
high_name_sim = 0
high_addr_sim = 0
either_high = 0

def clean_str(s):
    if not s or s is None: return ""
    s = unidecode(str(s).lower())
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return " ".join(s.split())

for _, row in df_pairs.iterrows():
    s1_n, s1_a = str(row['s1_name']), str(row['s1_addr'])
    m_n, m_a = str(row['m_name']), str(row['m_addr'])
    
    if s1_n == m_n: exact_name += 1
    if s1_a == m_a: exact_addr += 1
    
    c1_n = clean_str(s1_n)
    c2_n = clean_str(m_n)
    c1_a = clean_str(s1_a)
    c2_a = clean_str(m_a)
    
    if c1_n and c1_n == c2_n: norm_name_exact += 1
    if c1_a and c1_a == c2_a: norm_addr_exact += 1
    
    sim_name = fuzz.token_sort_ratio(c1_n, c2_n) if (c1_n and c2_n) else 0
    sim_addr = fuzz.token_sort_ratio(c1_a, c2_a) if (c1_a and c2_a) else 0
    
    if sim_name >= 80: high_name_sim += 1
    if sim_addr >= 80: high_addr_sim += 1
    if sim_name >= 80 or sim_addr >= 80: either_high += 1

n = len(df_pairs)
print(f"\n--- Signal Analysis on {n:,} True Positive Pairs ---")
print(f"Exact raw name match:            {exact_name:,} ({exact_name/n*100:.2f}%)")
print(f"Exact raw addr match:            {exact_addr:,} ({exact_addr/n*100:.2f}%)")
print(f"Normalized exact name match:     {norm_name_exact:,} ({norm_name_exact/n*100:.2f}%)")
print(f"Normalized exact addr match:     {norm_addr_exact:,} ({norm_addr_exact/n*100:.2f}%)")
print(f"High name sim (token_sort >= 80): {high_name_sim:,} ({high_name_sim/n*100:.2f}%)")
print(f"High addr sim (token_sort >= 80): {high_addr_sim:,} ({high_addr_sim/n*100:.2f}%)")
print(f"Either high sim (name OR addr):  {either_high:,} ({either_high/n*100:.2f}%)")
