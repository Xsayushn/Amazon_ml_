import zipfile
import io
import sys

sys.stdout.reconfigure(encoding='utf-8')
z = zipfile.ZipFile('6ab10eb3b23ba_student_resource.zip')

# Sample target IDs to inspect
target_s1 = 'S1-965667'
target_matches = {'S2-681193310', 'S2-743505751', 'S3-775321672', 'S3-11291185', 'S3-860443364'}

# Also pick another example from India
target_s1_b = 'S1-55344266'
target_matches_b = {'S2-249013014', 'S2-197070651', 'S3-478195123', 'S3-384364074'}

all_targets = {target_s1, target_s1_b} | target_matches | target_matches_b

records = {}

for name in ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv']:
    with z.open(f'student_resource/dataset/train/{name}') as f:
        reader = io.TextIOWrapper(f, encoding='utf-8')
        header = reader.readline().strip().split('\t')
        for line in reader:
            parts = line.strip().split('\t')
            if parts[0] in all_targets:
                records[parts[0]] = parts

print(f'=== Ground Truth Entity Group 1: {target_s1} ===')
for eid in [target_s1] + sorted(list(target_matches)):
    if eid in records:
        d = records[eid]
        print(f"[{eid}] Name: '{d[1]}' | Address: '{d[2]}' | Country: '{d[3]}'")

print(f'\n=== Ground Truth Entity Group 2: {target_s1_b} ===')
for eid in [target_s1_b] + sorted(list(target_matches_b)):
    if eid in records:
        d = records[eid]
        print(f"[{eid}] Name: '{d[1]}' | Address: '{d[2]}' | Country: '{d[3]}'")
