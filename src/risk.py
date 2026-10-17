# Risk measures: sample uniqueness, record linkage, attribute disclosure
import warnings

import numpy as np
import pandas as pd

from src.data import SEED, QI_EXT, NUMERIC_QIS
from src.hierarchies import HIERARCHIES, generalize
from src.numeric import to_numeric_view

# the attacker only knows LIMIT_BAL approximately: 50k bands (hierarchy level 1)
EXTERNAL_LEVELS = {"LIMIT_BAL": 1}


def combination_sizes(df, qis):
    # for each record, how many records share its QI combination
    return df.groupby(qis, observed=True, dropna=False)[qis[0]].transform("size")


def sample_uniqueness(df, qis):
    # share of records whose QI combination appears once in the sample
    return (combination_sizes(df, qis) == 1).mean()


def rare_combination_rate(df, qis, threshold):
    return (combination_sizes(df, qis) < threshold).mean()


def make_release_and_external(df, release_frac=0.8, external_frac=0.5, seed=SEED):
    # df is the whole population. The release is a random sample of it (the
    # microdata are a subset of the population). The attacker's dataset is an
    # independent sample of the population, so it also has people who are not
    # in the release. The row index is the identity of the person: it is only
    # used to check whether a link is correct, never to do the matching.
    rng = np.random.default_rng(seed)
    n = len(df)
    released = df.iloc[np.sort(rng.choice(n, round(release_frac * n), replace=False))]
    ext_rows = np.sort(rng.choice(n, round(external_frac * n), replace=False))
    external = generalize(df.iloc[ext_rows][QI_EXT], EXTERNAL_LEVELS)
    return released, external


def _to_level(s, col, level, target):
    # bring a column from hierarchy `level` up to `target` (bands are nested)
    if level >= target:
        return s
    if col in NUMERIC_QIS:
        s = to_numeric_view(s, "low").round()
    vals = s.fillna(0)
    if col in NUMERIC_QIS:
        vals = vals.astype(int)
    return HIERARCHIES[col][target](vals).where(s.notna())


def _is_tail(s):
    # values of a top/bottom coded column that are "<x" or ">x"
    return s.map(lambda v: isinstance(v, str) and v[0] in "<>")


def _apply_function_scheme(rel_s, ext_s, col, f, ext_lvl):
    # scheme given as a function f of the values (top/bottom coding): the
    # attacker applies f to his own column. If his column is in bands, a band
    # is a tail when f turns its mid point into "<x" / ">x"; the other values
    # are compared as bands, so the released values are put in the same bands.
    if ext_lvl == 0:
        return rel_s, f(ext_s)
    coded = f(to_numeric_view(ext_s))
    ext_out = ext_s.where(~_is_tail(coded), coded)
    tail_rel = _is_tail(rel_s)
    num = to_numeric_view(rel_s.where(~tail_rel))
    banded = HIERARCHIES[col][ext_lvl](num.round().fillna(0).astype(int)).where(num.notna())
    return banded.where(~tail_rel, rel_s), ext_out


def align(released, external, qis, scheme=None, numeric_mid=False):
    # puts released and external in the same domain, only for the comparison
    # (the released dataset itself is never modified).
    # What the attacker knows: if the release was protected with deterministic
    # techniques (generalization, recoding, top/bottom coding) he knows the
    # scheme and applies it to his own data: scheme = {column: hierarchy level
    # or function}. If it was protected with perturbative techniques (or local
    # suppression) he only has the original values: scheme=None.
    # In both cases the external LIMIT_BAL is in 50k bands, so released
    # LIMIT_BAL is brought to the same bands (or coarser, if the scheme is).
    # numeric_mid: numeric QIs are compared as numbers, bands -> mid point.
    scheme = scheme or {}
    rel, ext = released[qis].copy(), external[qis].copy()
    for col in qis:
        if numeric_mid and col in NUMERIC_QIS:
            rel[col], ext[col] = to_numeric_view(rel[col]), to_numeric_view(ext[col])
            continue
        ext_lvl = EXTERNAL_LEVELS.get(col, 0)
        rule = scheme.get(col, 0)
        if callable(rule):
            rel[col], ext[col] = _apply_function_scheme(rel[col], ext[col], col, rule, ext_lvl)
            continue
        target = max(rule, ext_lvl)
        rel[col] = _to_level(rel[col], col, rule, target)
        ext[col] = _to_level(ext[col], col, ext_lvl, target)
    return rel, ext


def record_linkage(released, external, qis, scheme=None):
    """Deterministic record linkage (course slides, record linkage: deterministic, slide 82).

    Exact join on the QIs. An external record is linked only if exactly one
    released record has the same QIs; it counts if that record is the right
    person (the row index is only used for this check).
    Returns linkage_rate = correctly linked external records / len(external).
    """
    rel, ext = align(released, external, qis, scheme)
    rel = rel.dropna().rename_axis("pid").reset_index()     # missing never matches
    rel = rel[~rel.duplicated(qis, keep=False)]             # unique matches only
    ext = ext.rename_axis("pid").reset_index()

    with warnings.catch_warnings():     # perturbed floats vs integer QIs: no match is expected
        warnings.simplefilter("ignore", UserWarning)
        linked = ext.merge(rel, on=qis, suffixes=("_ext", "_rel"))
    return (linked["pid_ext"] == linked["pid_rel"]).sum() / len(ext)


def record_linkage_distance(released, external, qis, stds, weights=None, scheme=None, block=1000, seed=SEED):
    """Distance-based record linkage (course slides, record linkage: distance-based, slide 83).

    Each external record is associated with the most similar released record.
    The course leaves the similarity function f free: the distance and the
    weights below are our design choice. Distance = weighted sum over the QIs of
      numeric: |a - b| / std of the original data (bands -> mid point)
      categorical: 0 if equal, 1 if different
      missing in either record (local suppression): 0.5
    Every external record is always associated with its nearest released
    record. If several released records tie at the minimum distance, one of
    them is picked at random (rng with `seed`). A rule like "link only if the
    nearest is unique" would make the attacker behave differently on original
    data (ties block the link) and on perturbed data (ties vanish, so he always
    guesses), which would bias the comparison between techniques.
    Returns linkage_rate = correctly linked external records / len(external).
    """
    rel, ext = align(released, external, qis, scheme, numeric_mid=True)
    w = {c: (weights or {}).get(c, 1.0) for c in qis}

    rel_cols, ext_cols = {}, {}
    for c in qis:
        if c in NUMERIC_QIS:
            rel_cols[c] = rel[c].to_numpy(np.float32) / np.float32(stds[c])
            ext_cols[c] = ext[c].to_numpy(np.float32) / np.float32(stds[c])
        else:   # categories -> integer codes shared by both sides, -1 = missing
            codes, _ = pd.factorize(pd.concat([rel[c], ext[c]]).astype(object))
            rel_cols[c], ext_cols[c] = codes[:len(rel)], codes[len(rel):]

    pos = rel.index.get_indexer(ext.index)      # position of the right person, -1 if not released
    rng = np.random.default_rng(seed)
    correct = 0
    for start in range(0, len(ext), block):
        sl = slice(start, start + block)
        dist = np.zeros((len(pos[sl]), len(rel)), dtype=np.float32)
        for c in qis:
            a, b = ext_cols[c][sl, None], rel_cols[c][None, :]
            if c in NUMERIC_QIS:
                d = np.abs(a - b)
                missing = np.isnan(a) | np.isnan(b)
            else:
                d = (a != b).astype(np.float32)
                missing = (a < 0) | (b < 0)
            dist += np.float32(w[c]) * np.where(missing, np.float32(0.5), d)

        ties = dist <= dist.min(axis=1, keepdims=True) + 1e-5
        # random choice among the tied records: argmax of a random score on the ties
        nearest = np.where(ties, rng.random(dist.shape, dtype=np.float32), -1).argmax(axis=1)
        correct += (nearest == pos[sl]).sum()

    return correct / len(ext)


def attribute_disclosure_rate(df, qis, target, external=None, scheme=None):
    """Attribute disclosure rate (slide 42): share of records that sit in a QI
    combination with >= 2 records and a constant target; missing is a value of
    its own.

    Computed in the attacker's domain: if `external` is given, the release is
    brought to the same domain as the external dataset with `align` (LIMIT_BAL
    in 50k bands, the scheme of the deterministic protection), only for the
    computation; `df` is not modified.
    Meaningful only for original and non-perturbative releases. With perturbative
    techniques the groups of the release no longer correspond to real people
    (noise makes every record unique so the measure falls to ~0, micro-aggregation
    creates artificial groups so it rises), hence the experiments report NaN.
    """
    if external is None:
        keys = df[qis]
    else:
        keys, _ = align(df, external, qis, scheme)
    groups = pd.concat([keys, df[target]], axis=1).groupby(qis, observed=True, dropna=False)[target]
    return ((groups.transform("size") >= 2) & (groups.transform("nunique") == 1)).mean()
