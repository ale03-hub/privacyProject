# Loading and cleaning the UCI "default of credit card clients" dataset
import os

import numpy as np
import pandas as pd

SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CSV_PATH = os.path.join(RAW_DIR, "credit_default.csv")
XLS_PATH = os.path.join(RAW_DIR, "default of credit card clients.xls")

TARGET = "default"

# quasi-identifiers
QI = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
# attacker who also knows the credit limit
QI_EXT = QI + ["LIMIT_BAL"]

# direct identifiers
IDENTIFIERS = ["ID", "customer_code", "phone"]

SEX_LABELS = {1: "male", 2: "female"}
EDUCATION_LABELS = {1: "graduate", 2: "university", 3: "high school", 4: "others"}
MARRIAGE_LABELS = {1: "married", 2: "single", 3: "others"}


def load():
    # use the cached csv if it's already there
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    if os.path.exists(XLS_PATH):
        # first row of the xls has generic names (X1, X2, ...), so use the second one
        df = pd.read_excel(XLS_PATH, header=1)
    else:
        # otherwise download it from ucimlrepo
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=350)
        df = pd.concat([data.data.features, data.data.targets], axis=1)
        df.insert(0, "ID", np.arange(1, len(df) + 1))

    # fix column names (they differ depending on the source)
    df = df.rename(columns={
        "PAY_1": "PAY_0",
        "default payment next month": TARGET,
        "Y": TARGET,
    })

    # save as csv so next time it's faster
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def clean(df):
    df = df.copy()
    # EDUCATION 0, 5, 6 and MARRIAGE 0 aren't documented,
    # so put them in "others"
    df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
    df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)
    return df


def add_identifiers(df, seed=SEED):
    # add fake customer codes and phone numbers to simulate a real release
    rng = np.random.default_rng(seed)
    n = len(df)
    df = df.copy()

    codes = rng.choice(10_000_000, n, replace=False)
    df["customer_code"] = ["C" + str(c).zfill(7) for c in codes]

    phones = rng.choice(100_000_000, n, replace=False)
    df["phone"] = ["09" + str(p).zfill(8) for p in phones]
    return df


def sanitize(df):
    # just drop the direct identifiers (not enough on its own)
    cols = [c for c in IDENTIFIERS if c in df.columns]
    return df.drop(columns=cols)


def load_clean(with_identifiers=False):
    df = clean(load())
    # identifiers are only needed for the linkage scenario
    if with_identifiers:
        return add_identifiers(df)
    return sanitize(df)