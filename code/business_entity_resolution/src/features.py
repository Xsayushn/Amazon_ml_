"""
Feature Extraction Engine for Pairwise Entity Resolution
Amazon ML Challenge 2026
"""

from typing import List
from rapidfuzz import fuzz

FEATURE_NAMES = [
    "name_sort", "name_set", "name_core_sort", "name_ratio", 
    "comp_exact", "comp_ratio", "name_jaccard", "name_overlap",
    "addr_sort", "addr_set", "addr_jaccard", "addr_overlap", 
    "postal_match", "digit_overlap", "addr_empty", "src_type", 
    "shared_keys", "max_sim", "min_sim", "mean_sim"
]

def extract_pairwise_features(e1: dict, e2: dict, shared_keys: int, cid: str) -> List[float]:
    """
    Extract high-dimensional similarity features between an S1 entity and candidate.
    """
    n1, n2 = e1["n_info"], e2["n_info"]
    a1, a2 = e1["a_info"], e2["a_info"]
    
    c1_clean, c2_clean = n1["clean"], n2["clean"]
    c1_core, c2_core = n1["core"], n2["core"]
    
    name_sort = fuzz.token_sort_ratio(c1_clean, c2_clean) if (c1_clean and c2_clean) else 0.0
    name_set = fuzz.token_set_ratio(c1_clean, c2_clean) if (c1_clean and c2_clean) else 0.0
    name_core_sort = fuzz.token_sort_ratio(c1_core, c2_core) if (c1_core and c2_core) else 0.0
    name_ratio = fuzz.ratio(c1_clean, c2_clean) if (c1_clean and c2_clean) else 0.0
    
    comp1, comp2 = n1["compressed"], n2["compressed"]
    comp_exact = 1.0 if (comp1 and comp1 == comp2) else 0.0
    comp_ratio = fuzz.ratio(comp1, comp2) if (comp1 and comp2) else 0.0
    
    # Token Jaccard and Overlap Coefficient
    t1_n, t2_n = set(n1["tokens"]), set(n2["tokens"])
    inter_n = len(t1_n & t2_n)
    name_jaccard = inter_n / len(t1_n | t2_n) if (t1_n | t2_n) else 0.0
    min_n = min(len(t1_n), len(t2_n))
    name_overlap = inter_n / min_n if min_n > 0 else 0.0
    
    # Address features
    addr1, addr2 = a1["clean"], a2["clean"]
    addr_sort = fuzz.token_sort_ratio(addr1, addr2) if (addr1 and addr2) else 0.0
    addr_set = fuzz.token_set_ratio(addr1, addr2) if (addr1 and addr2) else 0.0
    
    t1_a, t2_a = a1.get("token_set", set()), a2.get("token_set", set())
    inter_a = len(t1_a & t2_a)
    addr_jaccard = inter_a / len(t1_a | t2_a) if (t1_a | t2_a) else 0.0
    min_a = min(len(t1_a), len(t2_a))
    addr_overlap = inter_a / min_a if min_a > 0 else 0.0
    
    # Postal code match
    p1, p2 = a1["postal_code"], a2["postal_code"]
    if p1 and p2:
        postal_match = 1.0 if p1 == p2 else 0.0
    else:
        postal_match = -1.0
        
    # Digits overlap
    d1, d2 = a1["digits"], a2["digits"]
    digit_overlap = len(d1 & d2) / len(d1 | d2) if (d1 | d2) else -1.0
    
    addr_empty = 1.0 if not addr2 else 0.0
    src_type = 1.0 if cid.startswith("S2-") else 2.0
    
    max_sim = max(name_sort, addr_sort)
    min_sim = min(name_sort, addr_sort)
    mean_sim = (name_sort + addr_sort) / 2.0
    
    return [
        name_sort, name_set, name_core_sort, name_ratio, comp_exact, comp_ratio,
        name_jaccard, name_overlap,
        addr_sort, addr_set, addr_jaccard, addr_overlap, postal_match, digit_overlap,
        addr_empty, src_type, float(shared_keys), max_sim, min_sim, mean_sim
    ]
