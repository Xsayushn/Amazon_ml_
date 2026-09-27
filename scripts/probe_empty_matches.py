import duckdb
import lightgbm as lgb
import numpy as np
import sys

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalizer import normalize_business_name, normalize_business_address
from features import extract_pairwise_features

model = lgb.Booster(model_file="models/lgb_matcher.txt")
con = duckdb.connect()

# Find 10 S1 entities in US where candidate_entity_ids is not empty, but matched_entity_ids is empty!
con.execute("""
    CREATE TABLE probe AS
    SELECT m.source1_entity_id, c.candidate_entity_ids
    FROM read_csv('output/matching_results.tsv', delim='\t', header=true) m
    JOIN read_csv('output/candidate_pairs.tsv', delim='\t', header=true) c
      ON m.source1_entity_id = c.source1_entity_id
    WHERE (m.matched_entity_ids IS NULL OR m.matched_entity_ids = '')
      AND (c.candidate_entity_ids IS NOT NULL AND c.candidate_entity_ids != '')
    LIMIT 5;
""")

probe_list = con.execute("SELECT * FROM probe").fetchall()

for s1_id, cand_str in probe_list:
    cands = cand_str.split(',')
    # Load S1 record
    s1_row = con.execute(f"SELECT * FROM read_csv('student_resource/dataset/test/test_source1.tsv', delim='\\t', header=true) WHERE entity_id = '{s1_id}'").fetchone()
    print(f"\n=======================================================")
    print(f"S1: [{s1_row[0]}] Name: '{s1_row[1]}' | Addr: '{s1_row[2]}' | Country: '{s1_row[3]}'")
    
    # Load candidates
    s2_cands = [c for c in cands if c.startswith('S2-')]
    s3_cands = [c for c in cands if c.startswith('S3-')]
    s2_vals = ",".join(f"('{c}')" for c in s2_cands)
    s3_vals = ",".join(f"('{c}')" for c in s3_cands)
    query = []
    if s2_cands:
        query.append(f"SELECT * FROM read_csv('student_resource/dataset/test/test_source2.tsv', delim='\\t', header=true) WHERE entity_id IN (VALUES {s2_vals})")
    if s3_cands:
        query.append(f"SELECT * FROM read_csv('student_resource/dataset/test/test_source3.tsv', delim='\\t', header=true) WHERE entity_id IN (VALUES {s3_vals})")
    
    cand_rows = con.execute(" UNION ALL ".join(query)).fetchall()
    
    e1 = {
        'n_info': normalize_business_name(s1_row[1]),
        'a_info': normalize_business_address(s1_row[2], s1_row[3])
    }
    
    print(f"Candidates ({len(cand_rows)} loaded):")
    for r in cand_rows:
        cid = r[0]
        e2 = {
            'n_info': normalize_business_name(r[1]),
            'a_info': normalize_business_address(r[2], r[3])
        }
        feats = extract_pairwise_features(e1, e2, 1, cid)
        prob = model.predict(np.array([feats], dtype=np.float32))[0]
        print(f"  -> [{cid}] Prob: {prob:.4f} | Name: '{r[1]}' | Addr: '{r[2]}'")
