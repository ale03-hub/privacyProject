# Risk measures: equivalence classes, uniqueness, k and record linkage
from src.data import SEED


def class_sizes(df, qis):
    # for each record, how many records share its QI values
    return df.groupby(qis, observed=True, dropna=False)[qis[0]].transform("size")


def uniqueness(df, qis):
    return (class_sizes(df, qis) == 1).mean()


def small_class_rate(df, qis, k):
    return (class_sizes(df, qis) < k).mean()


def k_anonymity(df, qis):
    return int(class_sizes(df, qis).min())


def make_external(df, frac=0.5, seed=SEED):
    # the attacker's dataset: a sample of the same people with their QIs
    return df.sample(frac=frac, random_state=seed)


def linkage_rate(released, external, qis):
    # an external record is linked if its QIs match exactly one released
    # record. It counts only if that record is the right person: the row
    # index is used just to check this, not to do the matching.
    # external has to be generalized like released.
    rel = released[qis].copy()
    rel["rel_id"] = rel.index
    rel = rel[class_sizes(released, qis) == 1]

    ext = external[qis].copy()
    ext["ext_id"] = ext.index

    merged = ext.merge(rel, on=qis)
    correct = (merged["ext_id"] == merged["rel_id"]).sum()
    return correct / len(external)
