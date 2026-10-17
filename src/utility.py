# Utility measures: AUC of a classifier that predicts `default`, information loss
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import SEED, TARGET, CATEGORICAL
from src.numeric import to_labels, to_numeric_view


def split_indices(df, seed=SEED):
    # 70/30 stratified split, computed once on the original data and
    # reused for every protected version
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.3, stratify=df[TARGET], random_state=seed
    )
    return train_idx, test_idx


def evaluate_classifier(df, train_idx, test_idx, features=None):
    # logistic regression on all columns except the target. The encoding only
    # depends on the column type after protection: SEX / EDUCATION / MARRIAGE
    # and any non-numeric column (generalized) or column with missing values
    # are one-hot encoded (missing = its own category), the rest is scaled.
    # features: columns to use (default: all but the target)
    X = df.drop(columns=[TARGET]) if features is None else df[list(features)].copy()
    cat = [c for c in X.columns
           if c in CATEGORICAL or not is_numeric_dtype(X[c]) or X[c].isna().any()]
    num = [c for c in X.columns if c not in cat]
    X[cat] = X[cat].astype(str)

    model = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ])),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    model.fit(X.loc[train_idx], df.loc[train_idx, TARGET])

    proba = model.predict_proba(X.loc[test_idx])[:, 1]
    pred = model.predict(X.loc[test_idx])
    return {
        "auc": roc_auc_score(df.loc[test_idx, TARGET], proba),
        "accuracy": accuracy_score(df.loc[test_idx, TARGET], pred),
    }


def information_loss_continuous(original, protected, cols):
    # (slides 85-86) per column, on the numeric view of the protected values
    # (bands -> mid point, "<x" / ">x" -> x): MSE and MAE of the standardized
    # values (original std), relative change of mean and of variance.
    # Rows with a missing protected value are left out of all four measures and
    # their share is reported. Returns the averages over the columns.
    rows = []
    for c in cols:
        x = to_numeric_view(original[c])
        y = to_numeric_view(protected[c])
        ok = y.notna() & x.notna()
        x, y = x[ok], y[ok]
        err = (y - x) / original[c].std()
        rows.append({
            "mse": (err ** 2).mean(),
            "mae": err.abs().mean(),
            "mean_change": abs(y.mean() - x.mean()) / abs(x.mean()),
            "var_change": abs(y.var() - x.var()) / x.var(),
            "missing_share": 1 - ok.mean(),
        })
    return pd.DataFrame(rows).mean().to_dict()


def _entropy_bits(p):
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def information_loss_categorical(original, protected, cols, target):
    # (slides 85-86) per column, comparing values as category labels (strings;
    # generalized values are therefore always "different" from the original):
    #  - changed_share: share of records whose category changed (missing = changed)
    #  - table_distance: 1/2 * sum |p - q| between the relative joint frequencies
    #    of (original column, target) and (protected column, target)
    #  - cond_entropy: H(original | protected) in bits
    # Returns the averages over the columns.
    rows = []
    for c in cols:
        o = to_labels(original[c])
        p = to_labels(protected[c]).fillna("<missing>")
        t = original[target]
        orig_table = pd.crosstab(o, t, normalize=True)
        prot_table = pd.crosstab(p, t, normalize=True)
        labels = orig_table.index.union(prot_table.index)
        diff = (orig_table.reindex(labels, fill_value=0) - prot_table.reindex(labels, fill_value=0))
        joint = pd.crosstab(o, p, normalize=True)       # P(original, protected)
        rows.append({
            "changed_share": (o != p).mean(),
            "table_distance": 0.5 * diff.abs().to_numpy().sum(),
            # H(O | P) = H(O, P) - H(P)
            "cond_entropy": _entropy_bits(joint.to_numpy().ravel()) - _entropy_bits(joint.sum(axis=0).to_numpy()),
        })
    return pd.DataFrame(rows).mean().to_dict()


def information_loss_total(original, protected, cols=("SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL")):
    # One information loss for any configuration, always on the same columns so
    # that configurations are comparable. Per column a loss in [0, 1]: 0 = data
    # intact, 1 = no information ("no data").
    #  - categorical: H(original | protected) / H(original), in bits; protected
    #    values are compared as category labels, missing is a category of its own
    #  - numeric: MAE(original, protected) / MAE(original, mean of the original),
    #    protected read with to_numeric_view (bands -> mid point, "<x" / ">x" -> x),
    #    missing replaced by the original mean; clipped to 1
    # Returns the loss of each column and il_total, their mean.
    # The course defines utility as 1 / information loss; the R-U map uses
    # 1 - il_total because 1 / IL diverges for the original data (IL = 0).
    loss = {}
    for c in cols:
        if c in CATEGORICAL:
            o = to_labels(original[c])
            p = to_labels(protected[c]).fillna("<missing>")
            joint = pd.crosstab(o, p, normalize=True)
            h_cond = _entropy_bits(joint.to_numpy().ravel()) - _entropy_bits(joint.sum(axis=0).to_numpy())
            loss[c] = h_cond / _entropy_bits(joint.sum(axis=1).to_numpy())
        else:
            x = original[c].astype(float)
            y = to_numeric_view(protected[c]).fillna(x.mean())
            loss[c] = min(1.0, (y - x).abs().mean() / (x - x.mean()).abs().mean())
    loss["il_total"] = float(np.mean([loss[c] for c in cols]))
    return loss
