"""
Multi-Pass Inverted Index and Candidate Generation Engine
Amazon ML Challenge 2026: Business Entity Resolution
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

COMMON_WORDS = {
    "street", "road", "avenue", "lane", "drive", "near", "opposite", 
    "behind", "north", "south", "east", "west", "floor", "shop", "block",
    "the", "and", "for", "with", "from", "city", "state", "plaza"
}

def extract_blocking_keys(entry: dict) -> Set[str]:
    """
    Generate multi-pass blocking keys for a record:
    1. Informative Name tokens (len >= 3)
    2. Compressed name prefixes (5 and 7 chars)
    3. Postal/PIN codes
    4. Distinctive address tokens (len >= 6)
    5. Numeric building/shop/plot identifiers (len >= 3)
    """
    keys = []
    country = entry["country"]
    n_info = entry["n_info"]
    a_info = entry["a_info"]
    
    # Pass 1: Name tokens
    for token in n_info["core_tokens"]:
        if len(token) >= 3 and token not in COMMON_WORDS:
            keys.append(f"{country}:tok:{token}")
            
    # Pass 2: Compressed prefix
    comp = n_info["compressed"]
    if len(comp) >= 5:
        keys.append(f"{country}:c5:{comp[:5]}")
    if len(comp) >= 7:
        keys.append(f"{country}:c7:{comp[:7]}")
        
    # Pass 3: Postal / PIN code
    postal = a_info["postal_code"]
    if postal:
        keys.append(f"{country}:post:{postal}")
        
    # Pass 4: Distinctive address tokens
    for w in a_info["tokens"]:
        if len(w) >= 6 and not w.isdigit() and w not in COMMON_WORDS:
            keys.append(f"{country}:atok:{w}")
            
    # Pass 5: Numeric codes
    for d in a_info["digits"]:
        if len(d) >= 3:
            keys.append(f"{country}:num:{d}")

    return set(keys)


class MultiPassBlocker:
    """
    High-efficiency inverted index with adaptive frequency pruning.
    Designed to maximize recall while bounding candidate set size.
    """
    def __init__(self, max_key_frequency: int = 800):
        self.max_key_frequency = max_key_frequency
        self.index = defaultdict(list)
        self.filtered_index = {}

    def fit(self, target_records: Dict[str, dict]):
        """Build the inverted index from target records (Source 2 and 3)."""
        raw_index = defaultdict(list)
        for eid, entry in target_records.items():
            keys = extract_blocking_keys(entry)
            for k in keys:
                raw_index[k].append(eid)
                
        # Frequency pruning: prune non-informative keys that appear too frequently
        self.filtered_index = {
            k: v for k, v in raw_index.items() if len(v) <= self.max_key_frequency
        }

    def get_candidates(self, s1_entry: dict, max_candidates: int = 15) -> List[Tuple[str, int]]:
        """
        Retrieve and rank candidates for an S1 entity.
        Returns list of (candidate_id, shared_key_count).
        """
        keys = extract_blocking_keys(s1_entry)
        cand_counts = defaultdict(int)
        
        for k in keys:
            if k in self.filtered_index:
                for cid in self.filtered_index[k]:
                    cand_counts[cid] += 1
                    
        if not cand_counts:
            return []

        top_candidates = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:max_candidates]
        return top_candidates
