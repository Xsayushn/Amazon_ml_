import duckdb

con = duckdb.connect()
res = con.execute("""
    SELECT DISTINCT country, count(*)
    FROM read_csv('student_resource/dataset/test/test_source1.tsv', delim='\t', header=true)
    WHERE LOWER(country) LIKE '%france%'
    GROUP BY country;
""").fetchall()
print("France country variants:", res)

res_all = con.execute("""
    SELECT DISTINCT country, count(*)
    FROM read_csv('student_resource/dataset/test/test_source1.tsv', delim='\t', header=true)
    GROUP BY country;
""").fetchall()
print("All countries:", res_all)
