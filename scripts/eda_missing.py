import zipfile
import io
import sys

sys.stdout.reconfigure(encoding='utf-8')
z = zipfile.ZipFile('6ab10eb3b23ba_student_resource.zip')

for fname in ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv']:
    path = f"student_resource/dataset/train/{fname}"
    total = 0
    empty_name = 0
    empty_addr = 0
    empty_both = 0
    
    with z.open(path) as f:
        reader = io.TextIOWrapper(f, encoding='utf-8')
        next(reader)
        for line in reader:
            total += 1
            parts = line.rstrip('\n').split('\t')
            name = parts[1].strip() if len(parts) > 1 else ''
            addr = parts[2].strip() if len(parts) > 2 else ''
            if not name:
                empty_name += 1
            if not addr:
                empty_addr += 1
            if not name and not addr:
                empty_both += 1
            if total >= 200000:
                break
    print(f"[{fname} - 200k sample]")
    print(f"  Empty name:    {empty_name:,} ({empty_name/total*100:.2f}%)")
    print(f"  Empty address: {empty_addr:,} ({empty_addr/total*100:.2f}%)")
    print(f"  Empty both:    {empty_both:,} ({empty_both/total*100:.2f}%)")
