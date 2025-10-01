import pandas as pd
import os
import re
from thefuzz import process

def normalize_country(name: str) -> str:
    if pd.isna(name):
        return None
    name = name.lower().strip()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z\s]", "", name)   # remove punctuation
    name = re.sub(r"\s+", " ", name)       # collapse spaces
    return name.strip()

country_aliases = {
    # us and uk
    "usa": "united states",
    "us": "united states",
    "united states of america": "united states",
    "america": "united states",
    "u s a": "united states",

    "uk": "united kingdom",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "great britain": "united kingdom",
    "britain": "united kingdom",

    # korea
    "south korea": "korea republic of",
    "korea republic of": "korea republic of",
    "republic of korea": "korea republic of",
    "korea south": "korea republic of",
    "s korea": "korea republic of",
    "ro korea": "korea republic of",
    "korea, south": "korea republic of",

    "north korea": "korea democratic peoples republic of",
    "democratic peoples republic of korea": "korea democratic peoples republic of",
    "dprk": "korea democratic peoples republic of",

    # russia / soviet-era variants
    "russia": "russian federation",
    "russian federation": "russian federation",
    "russian fed": "russian federation",
    "russian federation (the)": "russian federation",

    # ivory coast
    "ivory coast": "cote divoire",
    "cote d ivoire": "cote divoire",
    "cote divoire": "cote divoire",
    "cotedivoire": "cote divoire",

    # guinea-bissau
    "guinea bissau": "guinea bissau",
    "guinea-bissau": "guinea bissau",
    "republic of guinea bissau": "guinea bissau",

    # belarus
    "belarus": "belarus",
    "belorussia": "belarus",
    "byelorussia": "belarus",
    "republic of belarus": "belarus",

    # eswatini / swaziland
    "eswatini": "eswatini",
    "kingdom of eswatini": "eswatini",
    "swaziland": "eswatini",
    "kingdom of swaziland": "eswatini",

    # czechia
    "czechia": "czech republic",
    "czech republic": "czech republic",

    # other common alternates / formal names (helpful general additions)
    "prc": "china",
    "people s republic of china": "china",
    "people s republic of china (china)": "china",
    "china": "china",

    "iran": "iran",
    "islamic republic of iran": "iran",
    "iran islamic republic of": "iran",

    "syrian arab republic": "syria",
    "syria": "syria",

    "viet nam": "vietnam",
    "vietnam": "vietnam",

    "moldova": "moldova",
    "republic of moldova": "moldova",
    "moldova republic of": "moldova",

    "bolivia": "bolivia",
    "bolivia plurinational state of": "bolivia",

    "vatican": "vatican city",
    "vatican city": "vatican city",
    "holy see": "vatican city",

    "turkiye": "turkey",
    "turkey": "turkey",

    "cape verde": "cape verde",
    "cabo verde": "cape verde",

    "congo republic": "congo republic",
    "congo democratic republic": "democratic republic of the congo",
    "democratic republic of the congo": "democratic republic of the congo",
    "dr congo": "democratic republic of the congo",

    # small/alternate common forms
    "slovakia": "slovakia",
    "slovak republic": "slovakia",

    "north macedonia": "north macedonia",
    "republic of north macedonia": "north macedonia",

    # handy abbreviations
    "uae": "united arab emirates",
    "u a e": "united arab emirates",
    "drc": "democratic republic of the congo",
    "ukraine": "ukraine",
}

def apply_alias(name):
    return country_aliases.get(name, name)

def fuzzy_merge(df_left,
                df_right,
                left_on='Country_clean',
                right_on='Country_clean',
                right_cols=None,
                threshold=90,
                return_matches=False):
    """
    Fuzzy-merge df_right into df_left.
    - right_cols: list of columns from df_right to bring in (excluding right_on). If None, bring all except right_on.
    - threshold: minimum similarity score to accept a match (0-100).
    - return_matches: if True, also returns the mapping dict (left_value -> matched_right_value or None).
    """
    if left_on not in df_left.columns:
        raise KeyError(f"left_on '{left_on}' not found in left dataframe")
    if right_on not in df_right.columns:
        raise KeyError(f"right_on '{right_on}' not found in right dataframe")

    # prepare right columns to merge
    if right_cols is None:
        right_cols = [c for c in df_right.columns if c != right_on]
    else:
        right_cols = [c for c in right_cols if c != right_on]

    # unique right keys to match against
    right_keys = df_right[right_on].dropna().astype(str).unique().tolist()

    # build mapping from left values -> best right key (or None)
    mapping = {}
    left_vals = df_left[left_on].dropna().astype(str).unique().tolist()
    for val in left_vals:
        match = process.extractOne(val, right_keys)
        if match is None:
            mapping[val] = None
        else:
            best, score = match
            mapping[val] = best if score >= threshold else None

    # apply mapping to create helper key column
    left = df_left.copy()
    left['_fuzzy_key'] = left[left_on].astype(str).map(mapping)

    # prepare right subset (one row per right_on)
    right_subset = df_right[[right_on] + right_cols].copy()
    right_subset = right_subset.drop_duplicates(subset=right_on)

    # merge on helper key -> right_on
    merged = pd.merge(left, right_subset, left_on='_fuzzy_key', right_on=right_on, how='left', suffixes=('', '_r'))

    # For any columns that exist in both df_left and df_right, prefer df_left values and fill missing with right
    for col in right_cols:
        right_col_r = col + '_r'
        if col in df_left.columns:
            # If left has column, fillna from right and drop helper
            if right_col_r in merged.columns:
                merged[col] = merged[col].fillna(merged[right_col_r])
                merged.drop(columns=[right_col_r], inplace=True, errors='ignore')
        else:
            # left doesn't have column: rename the imported right_col_r to col (or keep col if no suffix)
            if col in merged.columns and right_col_r in merged.columns:
                # unlikely but handle by filling then dropping suffix
                merged[col] = merged[col].fillna(merged[right_col_r])
                merged.drop(columns=[right_col_r], inplace=True, errors='ignore')
            elif right_col_r in merged.columns:
                merged.rename(columns={right_col_r: col}, inplace=True)

    # cleanup helper and duplicate country columns
    merged.drop(
        columns=['_fuzzy_key'] + [c for c in merged.columns if c.startswith('Country_clean_r')],
        inplace=True,
        errors='ignore'
    )

    if return_matches:
        return merged, mapping
    return merged

