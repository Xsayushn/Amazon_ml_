"""
Text Normalization and Multilingual Phonetic Processing Engine
Amazon ML Challenge 2026: Business Entity Resolution
"""

import re
import unicodedata
from unidecode import unidecode

# Legal Suffixes across US, India, and France
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", 
    "pvt", "private", "llp", "co", "company", "gmbh", "sarl", "sa", "sas", 
    "eurl", "partners", "enterprises", "services", "solutions", "holdings",
    "group", "associates", "industries", "agency", "tech", "technologies"
}

# Indic phonetic translations for legal terms and common business words
INDIC_PHONETIC_REPLACEMENTS = [
    (r"\bele[l]?pi\b", "llp"),
    (r"\bp+r+a+i*v+e*t+\b", "private"),
    (r"\bl+i+m+i+t+e*d+\b", "limited"),
    (r"\be+n+t+e*r+p+r+a+i*s+e*s*\b", "enterprises"),
    (r"\bk+o+m+p+a+n+[iy]\b", "company"),
    (r"\bp+h+u+d+\b", "food"),
    (r"\bh+o+t+e*l\b", "hotel"),
    (r"\bv+e+n+c+h+[au]r+s*\b", "ventures"),
    (r"\bs+e*r*v+i+[sc]+e*s*\b", "services"),
    (r"\bi+n+v+e*s+t+m+e*n+t+s*\b", "investments"),
]

# US State abbreviations to full names
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming"
}

# Indian State abbreviations
INDIA_STATES = {
    "tn": "tamil nadu", "up": "uttar pradesh", "mh": "maharashtra", "dl": "delhi",
    "ka": "karnataka", "wb": "west bengal", "gj": "gujarat", "rj": "rajasthan",
    "ap": "andhra pradesh", "ts": "telangana", "mp": "madhya pradesh", "kl": "kerala",
    "hr": "haryana", "pb": "punjab", "or": "odisha", "as": "assam", "br": "bihar",
    "jh": "jharkhand", "uk": "uttarakhand", "hp": "himachal pradesh", "ga": "goa"
}

# Street abbreviations across English and French
STREET_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "pl": "place", "sq": "square",
    "ct": "court", "ter": "terrace", "rte": "route", "bldg": "building",
    "apt": "apartment", "ste": "suite", "fl": "floor", "opp": "opposite",
    "nr": "near", "b/h": "behind"
}

DOMAIN_REGEX = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-\.]+)\.(?:com|org|net|in|fr|co|io|biz|info)$", re.IGNORECASE)
POSTAL_CODE_REGEX = re.compile(r"\b(\d{5,6})\b")


def normalize_business_name(name: str) -> dict:
    """
    Produce normalized variants of a business name:
    - cleaned_name: standardized, transliterated, lowercased, legal suffix mapped
    - stem_tokens: core words without legal suffixes
    - compressed_name: alphanumeric-only string (great for domain matches & typos)
    """
    if not name or name != name:
        return {"clean": "", "core": "", "compressed": "", "tokens": []}

    s = str(name).strip()

    # Check for domain format (e.g. maurewilliamscolombier.com)
    m = DOMAIN_REGEX.match(s)
    if m:
        s = m.group(1).replace("-", " ")

    is_non_ascii = any(ord(c) > 127 for c in s)

    # Transliterate Indic and accented characters to Latin phonetics
    s = unidecode(s).lower()
    if is_non_ascii:
        s = re.sub(r"(.)\1+", r"\1", s)
        # Handle phonetic eses -> ss
        s = re.sub(r"\beses\b", "ss", s)

    # Apply Indic phonetic normalization
    for pattern, repl in INDIC_PHONETIC_REPLACEMENTS:
        s = re.sub(pattern, repl, s)

    # Clean punctuation
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = s.split()

    # Separate core tokens from legal suffixes
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = tokens

    clean_str = " ".join(tokens)
    core_str = " ".join(core_tokens)
    compressed_str = "".join(core_tokens) if core_tokens else "".join(tokens)

    return {
        "clean": clean_str,
        "core": core_str,
        "compressed": compressed_str,
        "tokens": tokens,
        "core_tokens": core_tokens,
    }


def normalize_business_address(address: str, country: str = "") -> dict:
    """
    Normalize address fields:
    - Expand street abbreviations (Rd -> road, St -> street, etc.)
    - Expand country-specific state codes (NY -> New York, TN -> Tamil Nadu)
    - Extract postal/PIN codes
    - Extract landmark/building identifiers
    """
    if not address or address != address:
        return {"clean": "", "tokens": [], "token_set": set(), "postal_code": "", "digits": set()}

    s = str(address).strip()
    s = unidecode(s).lower()

    # Extract postal code (5 digits for US/France, 6 digits for India)
    postal_matches = POSTAL_CODE_REGEX.findall(s)
    postal_code = postal_matches[0] if postal_matches else ""

    # Clean punctuation, replace commas and periods with spaces
    s = re.sub(r"[,;:\.\(\)\/\-\#]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = s.split()

    expanded_tokens = []
    state_map = INDIA_STATES if country.lower() == "india" else (US_STATES if country.lower() == "us" else {})

    for t in tokens:
        if t in STREET_ABBREVIATIONS:
            expanded_tokens.append(STREET_ABBREVIATIONS[t])
        elif t in state_map:
            expanded_tokens.append(state_map[t])
        else:
            expanded_tokens.append(t)

    clean_addr = " ".join(expanded_tokens)
    digits = set(re.findall(r"\b\d+\b", clean_addr))

    return {
        "clean": clean_addr,
        "tokens": expanded_tokens,
        "token_set": set(expanded_tokens),
        "postal_code": postal_code,
        "digits": digits,
    }
