# Risk measures: sample uniqueness, record linkage, attribute disclosure
from src.data import SEED


def combination_sizes(df, qis):
    # for each record, how many records share its QI combination
    return df.groupby(qis, observed=True, dropna=False)[qis[0]].transform("size")


def sample_uniqueness(df, qis):
    # share of records whose QI combination appears once in the sample
    return (combination_sizes(df, qis) == 1).mean()


def rare_combination_rate(df, qis, threshold):
    return (combination_sizes(df, qis) < threshold).mean()


def make_external(df, frac=0.5, seed=SEED):
    # the attacker's dataset: a sample of the same people with their QIs
    return df.sample(frac=frac, random_state=seed)


def record_linkage(released, external, qis):
    # an external record is linked if its QIs match exactly one released
    # record. It counts only if that record is the right person: the row
    # index is used just to check this, not to do the matching.
    # external has to be generalized like released.
    rel = released[qis].copy()
    rel["rel_id"] = rel.index
    rel = rel[combination_sizes(released, qis) == 1]

    ext = external[qis].copy()
    ext["ext_id"] = ext.index

    merged = ext.merge(rel, on=qis)
    correct = (merged["ext_id"] == merged["rel_id"]).sum()
    return correct / len(external)


def record_linkage_distance(released, external, qis):
    # distance-based linkage: nearest released record on standardized,
    # weighted QIs; rate of external records linked to the right person
    raise NotImplementedError


def attribute_disclosure_rate(df, qis, target):
    # share of records in a QI group with >= 2 records and constant target
    raise NotImplementedError
