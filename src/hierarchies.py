# Generalization hierarchies for quasi-identifiers
# Level 0 = original values; each successive level is coarser.


def _identity(s):
    return s


def bands(width):
    # fixed-width bands, e.g. width 5 -> "20-24", "25-29", ...
    def f(s):
        lo = (s // width) * width
        return lo.astype(str) + "-" + (lo + width - 1).astype(str)
    return f


def _age_3(s):
    return s.map(lambda a: "<30" if a < 30 else ("30-50" if a <= 50 else ">50"))


def _education_2(s):
    # 1 graduate, 2 university -> higher; 3 high school, 4 others -> other
    return s.map(lambda e: "higher" if e in (1, 2) else "other")


def _marriage_2(s):
    # 1 married -> married; 2 single, 3 others -> not married
    return s.map(lambda m: "married" if m == 1 else "not married")


HIERARCHIES = {
    "AGE": [_identity, bands(5), bands(10), _age_3],
    "EDUCATION": [_identity, _education_2],
    "MARRIAGE": [_identity, _marriage_2],
    "LIMIT_BAL": [_identity, bands(50_000), bands(100_000), bands(200_000)],
}


def generalize(df, levels):
    # levels: {column: level}; columns not listed are left untouched
    out = df.copy()
    for col, lvl in levels.items():
        out[col] = HIERARCHIES[col][lvl](df[col])
    return out
