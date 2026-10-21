import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MAX_SAMPLE_ROWS = 5000
MAX_FEATURES_TESTED = 100
ID_UNIQUE_RATIO = 0.95
LEAK_SCORE = 0.95
LEAK_MIN_GAIN = 0.10

ID_NAME_PATTERN = re.compile(r"(^|[_\s-])(id|uuid|guid|index|idx)$", re.IGNORECASE)


def _encode_feature(series: pd.Series) -> np.ndarray:
    """Turn any column into a single numeric column a tree can split on."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        filled = series.fillna(series.median())
        return filled.to_numpy(dtype=float).reshape(-1, 1)

    codes, _ = pd.factorize(series.astype("string"), use_na_sentinel=True)
    return codes.astype(float).reshape(-1, 1)


def _single_feature_score(x, y, task_type):
    """Cross-validated score of a tiny tree using ONE feature to predict the target."""
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    if task_type == "classification":
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        scoring = "balanced_accuracy"
    else:
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        scoring = "r2"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(cross_val_score(model, x, y, cv=cv, scoring=scoring).mean())


def detect_leakage(df: pd.DataFrame, target: str, task_type: str) -> dict:
    """Flag columns that are likely IDs or that predict the target suspiciously well."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    data = df.dropna(subset=[target])
    sampled = False

    if len(data) > MAX_SAMPLE_ROWS:
        data = data.sample(MAX_SAMPLE_ROWS, random_state=42)
        sampled = True

    if len(data) < 10:
        return {"findings": [], "tested_columns": 0, "sampled": sampled,
                "note": "Too few labelled rows to test for leakage."}

    if task_type == "classification":
        y, _ = pd.factorize(data[target].astype("string"))
        # Balanced accuracy of always guessing one class is 1 / number of classes.
        # (Plain accuracy would hide leaks when the target is imbalanced.)
        baseline = 1.0 / max(len(np.unique(y)), 1)
    else:
        y = pd.to_numeric(data[target], errors="coerce")
        data = data.loc[y.notna()]
        y = y.loc[y.notna()].to_numpy()
        baseline = 0.0

    features = [c for c in data.columns if c != target]
    findings = []
    tested = 0

    for column in features[:MAX_FEATURES_TESTED]:
        series = data[column]
        non_null = series.dropna()

        if non_null.empty or non_null.nunique() <= 1:
            continue

        unique_ratio = non_null.nunique() / len(non_null)
        name_looks_like_id = bool(ID_NAME_PATTERN.search(str(column)))
        is_float = pd.api.types.is_float_dtype(series)

        # 1) Identifier columns: a unique value per row carries no real signal.
        if (unique_ratio >= ID_UNIQUE_RATIO and not is_float) or (
            name_looks_like_id and unique_ratio >= 0.9
        ):
            findings.append({
                "column": str(column),
                "kind": "Identifier",
                "severity": "medium",
                "detail": f"{unique_ratio * 100:.0f}% of values are unique. "
                          "IDs let a model memorise rows instead of learning "
                          "patterns. Drop this column before training.",
                "score": None,
            })
            continue

        # 2) Suspiciously strong single-feature predictor.
        try:
            score = _single_feature_score(_encode_feature(series), y, task_type)
        except ValueError:
            continue

        tested += 1
        gain = score - baseline

        if score >= LEAK_SCORE and gain >= LEAK_MIN_GAIN:
            metric = "balanced accuracy" if task_type == "classification" else "R²"
            findings.append({
                "column": str(column),
                "kind": "Possible leakage",
                "severity": "high",
                "detail": f"This one column alone predicts the target with "
                          f"{metric} {score:.2f} (baseline {baseline:.2f}). "
                          "It may be derived from the target or only known "
                          "after the outcome. Verify before using it.",
                "score": round(score, 3),
            })

    findings.sort(key=lambda f: (f["severity"] != "high", f["column"]))

    return {
        "findings": findings,
        "tested_columns": tested,
        "sampled": sampled,
        "note": None,
    }
