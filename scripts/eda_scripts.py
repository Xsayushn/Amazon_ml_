import zipfile
import io
import sys
import unicodedata
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
z = zipfile.ZipFile('6ab10eb3b23ba_student_resource.zip')

def get_scripts(text):
    scripts = set()
    for ch in text:
        name = unicodedata.name(ch, '')
        if 'TAMIL' in name: scripts.add('TAMIL')
        elif 'DEVANAGARI' in name: scripts.add('DEVANAGARI')
        elif 'TELUGU' in name: scripts.add('TELUGU')
        elif 'BENGALI' in name: scripts.add('BENGALI')
        elif 'GUJARATI' in name: scripts.add('GUJARATI')
        elif 'MALAYALAM' in name: scripts.add('MALAYALAM')
        elif 'KANNADA' in name: scripts.add('KANNADA')
        elif 'ARABIC' in name: scripts.add('ARABIC')
        elif 'LATIN' in name: scripts.add('LATIN')
    return scripts

for split in ['train', 'test']:
    for fname in [f'{split}_source1.tsv', f'{split}_source2.tsv']:
        path = f"student_resource/dataset/{split}/{fname}"
        script_counts = Counter()
        with z.open(path) as f:
            reader = io.TextIOWrapper(f, encoding='utf-8')
            next(reader)
            for i, line in enumerate(reader):
                parts = line.rstrip('\n').split('\t')
                name = parts[1] if len(parts) > 1 else ''
                addr = parts[2] if len(parts) > 2 else ''
                s = get_scripts(name + ' ' + addr)
                for sc in s:
                    script_counts[sc] += 1
                if i >= 50000:
                    break
        print(f"[{fname} 50k sample] Scripts:", dict(script_counts))
