import zipfile
import io
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')
z = zipfile.ZipFile('6ab10eb3b23ba_student_resource.zip')

# Check how often non-latin names have matching addresses or tokens
print("Analyzing non-Latin matches...")
# Let's inspect 10 ground truth cases where name contains non-ascii characters
# We can load a small chunk of S1, S2, GT
gt_map = {}
with z.open('student_resource/dataset/train/train_ground_truth.tsv') as f:
    reader = io.TextIOWrapper(f, encoding='utf-8')
    next(reader)
    for i, line in enumerate(reader):
        s1, tab, rest = line.partition('\t')
        mids = rest.strip().split(',') if rest.strip() else []
        if mids:
            gt_map[s1] = mids
        if len(gt_map) >= 10000:
            break

s1_data = {}
with z.open('student_resource/dataset/train/train_source1.tsv') as f:
    reader = io.TextIOWrapper(f, encoding='utf-8')
    next(reader)
    for line in reader:
        parts = line.strip().split('\t')
        if parts[0] in gt_map:
            s1_data[parts[0]] = parts
        if len(s1_data) == len(gt_map):
            break

s2_data = {}
all_mids = set(m for mids in gt_map.values() for m in mids if m.startswith('S2-'))
with z.open('student_resource/dataset/train/train_source2.tsv') as f:
    reader = io.TextIOWrapper(f, encoding='utf-8')
    next(reader)
    for line in reader:
        parts = line.strip().split('\t')
        if parts[0] in all_mids:
            s2_data[parts[0]] = parts
        if len(s2_data) >= 15000:
            break

found = 0
for s1, mids in gt_map.items():
    if s1 not in s1_data: continue
    for m in mids:
        if m in s2_data:
            s2_row = s2_data[m]
            name2 = s2_row[1]
            if any(ord(c) > 127 for c in name2):
                found += 1
                print(f"--- Example {found} ---")
                print(f"S1: Name: '{s1_data[s1][1]}' | Addr: '{s1_data[s1][2]}'")
                print(f"S2: Name: '{name2}' | Addr: '{s2_row[2]}'")
                if found >= 5:
                    break
    if found >= 5:
        break
