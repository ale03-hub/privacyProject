# Generalization hierarchies for quasi-identifiers
# Level 0 = original values; each successive level is coarser.
# The last level of every hierarchy is "*" (the attribute is fully suppressed).


def _identity(s):
    return s


def _star(s):
    return s.map(lambda _: "*")


def _bands(width):
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


HIERARCHIES = {
    "AGE": [_identity, _bands(5), _bands(10), _age_3, _star],
    "EDUCATION": [_identity, _education_2, _star],
    "MARRIAGE": [_identity, _star],
    "SEX": [_identity, _star],
    "LIMIT_BAL": [_identity, _bands(50_000), _bands(100_000), _bands(200_000), _star],
}


def max_levels(qis):
    # highest generalization level available for each quasi-identifier
    return {q: len(HIERARCHIES[q]) - 1 for q in qis}


def generalize(df, levels):
    # levels: {column: level}; columns not listed are left untouched
    out = df.copy()
    for col, lvl in levels.items():
        out[col] = HIERARCHIES[col][lvl](df[col])
    return out
