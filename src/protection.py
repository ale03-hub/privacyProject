# Masking: non-perturbative and perturbative techniques.
# Every function takes a DataFrame and returns a protected copy (the input is
# never modified). Randomness always comes from a generator seeded with `seed`.
# (Generalization by hierarchy level: see hierarchies.generalize)
import numpy as np

from src.data import SEED
from src.hierarchies import bands
from src.numeric import to_labels


# --- non-perturbative ---------------------------------------------------

def global_recoding(df, col, width):
    # (slide 56) replace a numeric column with fixed-width bands "a-b"
    out = df.copy()
    out[col] = bands(width)(df[col]).where(df[col].notna())
    return out


def top_bottom_series(s, lower=None, upper=None):
    # values < lower -> "<lower", > upper -> ">upper", the others as text
    out = to_labels(s)
    if lower is not None:
        out = out.where(~(s < lower), f"<{lower}")
    if upper is not None:
        out = out.where(~(s > upper), f">{upper}")
    return out


def top_bottom_coding(df, col, lower=None, upper=None):
    # (slides 57-58) the column becomes categorical: "<lower", ">upper" or the value
    out = df.copy()
    out[col] = top_bottom_series(df[col], lower, upper)
    return out


def local_suppression(df, qis, threshold):
    # (slide 55) while some records are in a QI combination with frequency <
    # threshold, set to missing one more QI of those records: the one with most
    # distinct values among those not yet suppressed (so AGE or LIMIT_BAL first).
    # Missing counts as a value of its own when counting combinations.
    # Returns the protected frame and the % of QI cells suppressed. A record
    # with all QIs suppressed is left alone, even if the all-missing combination
    # is itself smaller than the threshold.
    out = df.copy()
    order = sorted(qis, key=lambda c: -df[c].nunique())
    while True:
        sizes = out.groupby(qis, observed=True, dropna=False)[qis[0]].transform("size")
        todo = (sizes < threshold).to_numpy().copy()
        if not todo.any():
            break
        suppressed = False
        for c in order:
            hit = todo & out[c].notna().to_numpy()
            out[c] = out[c].where(~hit)
            suppressed |= hit.any()
            todo &= ~hit
        if not suppressed:      # the rare records have no QI left to suppress
            break
    pct = 100 * (out[qis].isna().sum().sum() - df[qis].isna().sum().sum()) / (len(df) * len(qis))
    return out, pct


# --- perturbative -------------------------------------------------------

def pram(df, col, p_keep, seed=SEED, transition=None):
    # (slides 65-67) every value stays with probability p_keep, otherwise it
    # moves to another category with probability proportional to the frequency
    # of that category. `transition` (matrix, rows = original category, columns
    # = new category, both in sorted category order; DataFrame or array)
    # replaces this construction.
    out = df.copy()
    values = df[col]
    cats = np.sort(values.dropna().unique())
    k = len(cats)
    freq = values.value_counts(normalize=True).reindex(cats).to_numpy()
    if transition is None:
        others = freq[None, :] * (1 - np.eye(k))
        P = np.eye(k) * p_keep + (1 - p_keep) * others / others.sum(axis=1, keepdims=True)
    else:
        P = np.asarray(transition.loc[cats, cats] if hasattr(transition, "loc") else transition, float)
    assert P.shape == (k, k) and np.allclose(P.sum(axis=1), 1), "transition rows must sum to 1"

    rng = np.random.default_rng(seed)
    mask = values.notna().to_numpy()
    codes = np.searchsorted(cats, values[mask].to_numpy())
    new = (rng.random(len(codes))[:, None] >= np.cumsum(P, axis=1)[codes]).sum(axis=1)
    out.loc[mask, col] = cats[np.minimum(new, k - 1)]
    return out


def additive_noise(df, col, sigma, seed=SEED):
    # (slide 68) uncorrelated noise N(0, (sigma * std)^2); AGE is rounded to an
    # integer; values are clipped at the minimum observed in the column
    out = df.copy()
    rng = np.random.default_rng(seed)
    x = df[col].astype(float)
    y = (x + rng.normal(0, sigma * x.std(), len(x))).clip(lower=x.min())
    out[col] = y.round().astype(df[col].dtype) if col == "AGE" else y
    return out


def rank_swapping(df, col, p, seed=SEED):
    # (slide 70) sort by value; every record not yet swapped is swapped with a
    # random not yet swapped record whose rank is at most p% of n away.
    # p is a percentage (10 = 10%). The marginal distribution is preserved.
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = df.index[df[col].notna()]
    values = df.loc[idx, col].to_numpy()
    order = np.argsort(values, kind="stable")
    n = len(order)
    window = max(1, round(p / 100 * n))
    swapped = np.zeros(n, dtype=bool)
    new = values.copy()
    for i in range(n):
        if swapped[i]:
            continue
        # earlier ranks are all swapped or had no free partner: look upwards
        free = np.flatnonzero(~swapped[i + 1:i + window + 1])
        if len(free) == 0:
            continue
        j = i + 1 + rng.choice(free)
        swapped[i] = swapped[j] = True
        new[order[i]], new[order[j]] = values[order[j]], values[order[i]]
    out.loc[idx, col] = new
    return out


def microaggregation(df, cols, size):
    # (slide 71) standardize cols, sort the records on the first principal
    # component, form consecutive groups of `size` records (the last group
    # takes the remainder) and replace the values with the group mean (AGE
    # rounded to an integer). Simplified version (1D projection) of the
    # clustering of maximally similar records.
    out = df.copy()
    X = df[cols].astype(float)
    Z = ((X - X.mean()) / X.std()).to_numpy()
    first_pc = np.linalg.svd(Z, full_matrices=False)[2][0]
    rank = np.empty(len(df), dtype=int)
    rank[np.argsort(Z @ first_pc, kind="stable")] = np.arange(len(df))
    group = np.minimum(rank // size, max(len(df) // size - 1, 0))
    for c in cols:
        mean = X[c].groupby(group).transform("mean")
        out[c] = mean.round().astype(df[c].dtype) if c == "AGE" else mean
    return out
