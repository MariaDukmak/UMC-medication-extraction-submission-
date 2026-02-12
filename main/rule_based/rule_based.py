import re
import pandas as pd
from typing import Tuple, Optional


def parse_gebruiksvoorschrift(code: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse structured prescription codes like '1D1T', '3-6D1DR', '0.5W1T', including ranges and decimals.
    Multiple codes in one string are combined to get total min/max.

    Returns:
        (min, max) daily usage as float or int.
    """
    if not isinstance(code, str):
        return (None, None)

    s = code.upper()
    # Pattern: number[-number][D|W] followed by optional suffix
    pattern = r'(?<!\d)(\d+(?:[.,]\d+)?)(?:\s*-\s*(\d+(?:[.,]\d+)?))?\s*[DW][A-Z0-9-]*'
    hits = re.findall(pattern, s)
    if not hits:
        return (None, None)

    vals = []
    for lo, hi in hits:
        lo_v = float(lo.replace(',', '.'))
        hi_v = float(hi.replace(',', '.')) if hi else lo_v
        vals.extend([lo_v, hi_v])

    mn, mx = min(vals), max(vals)
    mn = int(mn) if float(mn).is_integer() else mn
    mx = int(mx) if float(mx).is_integer() else mx
    return (mn, mx)


def parse_vrije_tekst_gebruik(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse free-text usage like:
    - '1-2 tablets per day'
    - '1 to 3x per day'
    - '0.5-1 /dag' or '0.5 per week'
    Returns (min, max) or (None, None) if not found.
    """
    if not isinstance(text, str):
        return (None, None)

    s = text.lower()
    pattern = (
        r'(?<!\d)(\d+(?:[.,]\d+)?)(?:\s*(?:-|–|tot)\s*(\d+(?:[.,]\d+)?))?'
        r'(?:\s*x)?\s*(?:per\s+dag|/dag|dagelijks|x\s*/\s*dag|per\s*week|/week)'
    )
    m = re.search(pattern, s)
    if not m:
        return (None, None)

    def to_num(x: str) -> float:
        v = float(x.replace(',', '.'))
        return int(v) if v.is_integer() else v

    mn = to_num(m.group(1))
    mx = to_num(m.group(2)) if m.group(2) else mn
    return (mn, mx)


def merge_usages(t1: Tuple[Optional[float], Optional[float]],
                 t2: Tuple[Optional[float], Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Combine two (min,max) tuples to get the overall (min,max).
    """
    mins = [v for v in (t1[0], t2[0]) if v is not None]
    maxs = [v for v in (t1[1], t2[1]) if v is not None]
    if not mins and not maxs:
        return (None, None)
    mn = min(mins) if mins else None
    mx = max(maxs) if maxs else None
    if mn is None and mx is not None:
        mn = mx
    if mx is None and mn is not None:
        mx = mn
    return (mn, mx)


def get_strict_label(text: str) -> str:
    """Detect whether medication is 'as needed' or strict."""
    if isinstance(text, str) and re.search(r'\b(zo nodig|indien nodig|zn)\b', text.lower()):
        return 'not strict'
    return 'strict'


def extract_unit(sterkte: str) -> Optional[str]:
    """Extract standardized unit from product strength."""
    if not isinstance(sterkte, str):
        return None
    s = sterkte.upper().replace(" ", "")
    units = ['MG/G', 'UG/DO', 'UG/OD', 'MG/ML', 'IE/ML', 'UG', 'MG', 'G']
    for u in units:
        if u in s:
            return u
    return None


def get_duration_by_usage(total: float, gebruik: Optional[float]) -> Optional[float]:
    """Calculate duration in days from total quantity and daily usage."""
    try:
        if pd.isna(total) or gebruik in (None, 0) or pd.isna(gebruik):
            return None
        return total / gebruik
    except Exception:
        return None


def get_duration_by_dates(start, end) -> Optional[int]:
    """Calculate number of days between start and end dates."""
    try:
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        return (e - s).days
    except Exception:
        return None


def extract_vrije_label(text: str) -> Optional[str]:
    """
    Extract free-text comments, filtering out general tips.
    """
    if not isinstance(text, str):
        return None
    blacklist = {
        '1 maal per dag', '2 maal per dag', 'kuur afmaken',
        'pas op met alcohol', 'pas op met grapefruit',
        'voor de nacht', 'dagelijks', 'zo nodig', 'indien nodig'
    }
    parts = [p.strip() for p in re.split(r'[.,;]', text) if p.strip()]
    filtered = [p for p in parts if not any(b in p.lower() for b in blacklist)]
    return '; '.join(filtered) if filtered else None


def extract_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create structured labels from prescription DataFrame.

    Expected columns:
    ['id','Voorschrijfdatum','Einddatum','Gebruiksvoorschrift',
     'Hoeveelheid','Product_sterkte','Vrije_tekst',
     'Huisarts_id_umc']
    """
    original_cols = [
        'id', 'Voorschrijfdatum', 'Einddatum', 'Gebruiksvoorschrift',
        'Hoeveelheid', 'Product_sterkte', 'Vrije_tekst',
        'Huisarts_id_umc'
    ]

    df = df.copy()
    df_base = df[original_cols].copy()

    # Parse structured codes and free-text usage
    parsed_gv = df_base['Gebruiksvoorschrift'].apply(parse_gebruiksvoorschrift)
    parsed_vt = df_base['Vrije_tekst'].apply(parse_vrije_tekst_gebruik)

    # Combine to get final min/max usage
    merged = [merge_usages(gv, vt) for gv, vt in zip(parsed_gv, parsed_vt)]
    df_base['minimum_gebruik_label'] = [m[0] for m in merged]
    df_base['maximum_gebruik_label'] = [m[1] for m in merged]

    # Other labels
    df_base['strict_label'] = df_base['Vrije_tekst'].apply(get_strict_label)
    df_base['unit_label'] = df_base['Product_sterkte'].apply(extract_unit)

    # Duration based on min/max usage
    df_base['duration_by_min_usage'] = [
        get_duration_by_usage(h, u)
        for h, u in zip(df_base['Hoeveelheid'], df_base['minimum_gebruik_label'])
    ]
    df_base['duration_by_max_usage'] = [
        get_duration_by_usage(h, u)
        for h, u in zip(df_base['Hoeveelheid'], df_base['maximum_gebruik_label'])
    ]

    # Duration based on start/end dates
    df_base['duration_by_dates'] = [
        get_duration_by_dates(s, e)
        for s, e in zip(df_base['Voorschrijfdatum'], df_base['Einddatum'])
    ]

    # Extract free-text comments
    df_base['vrije_label'] = df_base['Vrije_tekst'].apply(extract_vrije_label)

    return df_base
