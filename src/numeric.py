# Helpers to read a column that may have been protected (bands, top/bottom
# coding, missing values) either as numbers or as category labels
import pandas as pd


def to_numeric_view(s, how="mid"):
    # numbers, whatever the protection: "a-b" -> mid point (or `a` if how="low"),
    # "<x" / ">x" -> x, numeric strings -> number, missing -> NaN
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def f(v):
        if not isinstance(v, str):
            return v
        if v[0] in "<>":
            return float(v[1:])
        if "-" in v[1:]:
            lo, hi = _split_range(v)
            return (lo + hi) / 2 if how == "mid" else lo
        return float(v)

    return pd.to_numeric(s.map(f))


def _split_range(v):
    # "20-24" -> (20, 24); the first character may be a minus sign
    head, tail = v[1:].split("-", 1)
    return float(v[0] + head), float(tail)


def to_labels(s):
    # category labels as strings, with 24.0 -> "24"; missing stays NaN
    def f(v):
        if pd.isna(v):
            return v
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    return s.map(f)
