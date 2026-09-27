import duckdb

con = duckdb.connect()

con.execute("""
    CREATE TABLE sample_s1 AS 
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true)
    LIMIT 10;
""")

con.execute("""
    CREATE TABLE sample_gt AS
    SELECT g.source1_entity_id, g.matched_entity_ids
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true) g
    JOIN sample_s1 s ON g.source1_entity_id = s.entity_id;
""")

pairs = con.execute("""
    SELECT source1_entity_id, UNNEST(string_split(CAST(matched_entity_ids AS VARCHAR), ',')) as matched_id
    FROM sample_gt
    WHERE matched_entity_ids IS NOT NULL;
""").fetchall()

s1_dict = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT * FROM sample_s1").fetchall()}

matched_ids = [p[1] for p in pairs if p[1]]
con.execute(f"""
    CREATE TABLE target_records AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM (VALUES {','.join(f"('{m}')" for m in matched_ids if m.startswith('S2-'))}) as t(matched_id))
    UNION ALL
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
    WHERE entity_id IN (SELECT matched_id FROM (VALUES {','.join(f"('{m}')" for m in matched_ids if m.startswith('S3-'))}) as t(matched_id));
""")

t_dict = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT * FROM target_records").fetchall()}

for s1_id in s1_dict:
    print(f"\n=======================================================")
    print(f"S1 Entity: {s1_id}")
    print(f"  Name:    '{s1_dict[s1_id][0]}'")
    print(f"  Address: '{s1_dict[s1_id][1]}'")
    print(f"  Country: '{s1_dict[s1_id][2]}'")
    print(f"Ground Truth Matches:")
    for p in pairs:
        if p[0] == s1_id and p[1] in t_dict:
            print(f"  -> [{p[1]}] Name: '{t_dict[p[1]][0]}' | Addr: '{t_dict[p[1]][1]}'")
