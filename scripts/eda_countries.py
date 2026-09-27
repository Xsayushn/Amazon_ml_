import zipfile
import io
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
z = zipfile.ZipFile('6ab10eb3b23ba_student_resource.zip')

for split in ['train', 'test']:
    for src in ['source1', 'source2', 'source3']:
        fname = f"{split}_{src}.tsv"
        path = f"student_resource/dataset/{split}/{fname}"
        counts = Counter()
        with z.open(path) as f:
            reader = io.TextIOWrapper(f, encoding='utf-8')
            next(reader)
            for line in reader:
                parts = line.strip().split('\t')
                c = parts[3] if len(parts) >= 4 else '<MISSING>'
                counts[c] += 1
        print(f"Country counts in {fname}:", dict(counts))
