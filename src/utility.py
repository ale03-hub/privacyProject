# Utility measures: AUC of a classifier that predicts `default`
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import SEED, TARGET


def split_indices(df, seed=SEED):
    # 70/30 stratified split, computed once on the original data and
    # reused for every protected version
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.3, stratify=df[TARGET], random_state=seed
    )
    return train_idx, test_idx


def evaluate_classifier(df, qis, train_idx, test_idx):
    # logistic regression on all columns except the target:
    # QIs are one-hot encoded (they may be generalized strings), the rest is scaled
    # suppressed records are missing from df, so keep only the indices that are left
    train_idx = df.index.intersection(train_idx)
    test_idx = df.index.intersection(test_idx)

    other = [c for c in df.columns if c not in qis and c != TARGET]
    model = Pipeline([
        ("prep", ColumnTransformer([
            ("qi", OneHotEncoder(handle_unknown="ignore"), qis),
            ("num", StandardScaler(), other),
        ])),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])

    X = df.drop(columns=[TARGET])
    # one-hot needs the QIs to be treated as categories
    X[qis] = X[qis].astype(str)
    model.fit(X.loc[train_idx], df.loc[train_idx, TARGET])

    proba = model.predict_proba(X.loc[test_idx])[:, 1]
    pred = model.predict(X.loc[test_idx])
    return {
        "auc": roc_auc_score(df.loc[test_idx, TARGET], proba),
        "accuracy": accuracy_score(df.loc[test_idx, TARGET], pred),
    }
