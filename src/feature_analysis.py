import numpy as np
import pandas as pd

MAX_NUMERIC_COLUMNS = 100
HIGH_CARDINALITY_MIN_UNIQUE = 50
CORRELATION_THRESHOLD = 0.90
SKEW_THRESHOLD = 2.0
OUTLIER_PERCENT_THRESHOLD = 5.0
MIXED_TYPE_MIN_SHARE = 0.80


def _numeric_feature_columns(df: pd.DataFrame, exclude: set) -> list:
    """Numeric, non-boolean columns that vary and have more than a handful of values."""
    columns = []
    for column in df.columns:
        if column in exclude:
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            continue
        if series.nunique(dropna=True) > 10:
            columns.append(column)
    return columns[:MAX_NUMERIC_COLUMNS]


def analyze_features(df: pd.DataFrame, target: str = None) -> dict:
    """Check feature columns for problems that make modelling harder."""
    exclude = {target} if target else set()
    feature_columns = [c for c in df.columns if c not in exclude]
    n_rows = len(df)
    n_features = len(feature_columns)

    issues = []

    # ---- Dataset size relative to the number of features ----
    rows_per_feature = round(n_rows / max(n_features, 1), 1)

    if n_rows < 100:
        issues.append({"severity": "high",
                       "message": f"Only {n_rows} rows. Most models need far more "
                                  "data, and validation scores will be very noisy."})
    elif n_rows < 500:
        issues.append({"severity": "medium",
                       "message": f"Only {n_rows} rows. Results will vary a lot "
                                  "between splits; prefer cross-validation."})

    if rows_per_feature < 10:
        issues.append({"severity": "medium",
                       "message": f"Just {rows_per_feature} rows per feature. "
                                  "Models can overfit when features are this many "
                                  "relative to rows."})

    # ---- Text columns that are really numbers (or mostly numbers) ----
    mixed_types = []
    text_numbers = []
    for column in feature_columns:
        series = df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        share = float(pd.to_numeric(non_null, errors="coerce").notna().mean())
        if share == 1.0:
            text_numbers.append({"column": str(column)})
        elif share >= MIXED_TYPE_MIN_SHARE:
            mixed_types.append({"column": str(column),
                                "numeric_share": round(share * 100, 1)})

    for item in text_numbers:
        issues.append({"severity": "medium",
                       "message": f"'{item['column']}' holds numbers stored as text. "
                                  "Convert it to a numeric type."})
    for item in mixed_types:
        issues.append({"severity": "high",
                       "message": f"'{item['column']}' mixes numbers and text "
                                  f"({item['numeric_share']}% numeric). Find and "
                                  "clean the non-numeric entries."})

    # ---- High-cardinality categoricals ----
    high_cardinality = []
    already_flagged = {m["column"] for m in mixed_types} | {t["column"] for t in text_numbers}
    for column in feature_columns:
        if str(column) in already_flagged:
            continue  # a number-like text column is a type problem, not a category problem
        series = df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) \
                or isinstance(series.dtype, pd.CategoricalDtype):
            unique = int(series.nunique(dropna=True))
            # Near-unique columns are reported by the identifier check instead.
            if unique >= HIGH_CARDINALITY_MIN_UNIQUE and unique / max(n_rows, 1) < 0.95:
                high_cardinality.append({"column": str(column), "unique": unique})

    for item in high_cardinality:
        issues.append({"severity": "medium",
                       "message": f"'{item['column']}' has {item['unique']} distinct "
                                  "categories. One-hot encoding will explode the "
                                  "feature count; consider grouping or target encoding."})

    numeric_columns = _numeric_feature_columns(df, exclude)

    # ---- Highly correlated numeric pairs ----
    correlated_pairs = []
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr(numeric_only=True).abs()
        values = corr.to_numpy()
        names = list(corr.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                value = values[i, j]
                if not np.isnan(value) and value >= CORRELATION_THRESHOLD:
                    correlated_pairs.append({"a": str(names[i]), "b": str(names[j]),
                                             "correlation": round(float(value), 3)})
        correlated_pairs.sort(key=lambda p: p["correlation"], reverse=True)
        correlated_pairs = correlated_pairs[:10]

    for pair in correlated_pairs:
        issues.append({"severity": "low",
                       "message": f"'{pair['a']}' and '{pair['b']}' are almost the "
                                  f"same signal (|r| = {pair['correlation']}). "
                                  "Keep one to reduce redundancy."})

    # ---- Skewed numeric columns and outliers ----
    skewed = []
    outliers = []
    for column in numeric_columns:
        values = df[column].dropna()
        if len(values) < 20:
            continue

        skew = float(values.skew())
        if abs(skew) >= SKEW_THRESHOLD:
            skewed.append({"column": str(column), "skew": round(skew, 2)})

        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            share = float(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean() * 100)
            if share >= OUTLIER_PERCENT_THRESHOLD:
                outliers.append({"column": str(column), "percentage": round(share, 1)})

    skewed.sort(key=lambda s: abs(s["skew"]), reverse=True)
    outliers.sort(key=lambda o: o["percentage"], reverse=True)

    for item in skewed[:5]:
        issues.append({"severity": "low",
                       "message": f"'{item['column']}' is heavily skewed "
                                  f"(skew {item['skew']}). A log transform may help "
                                  "linear and neural models."})
    for item in outliers[:5]:
        issues.append({"severity": "low",
                       "message": f"'{item['column']}' has {item['percentage']}% "
                                  "outliers by the IQR rule. Check whether they are "
                                  "errors or real extremes."})

    return {
        "rows": int(n_rows),
        "features": int(n_features),
        "rows_per_feature": rows_per_feature,
        "numeric_tested": len(numeric_columns),
        "high_cardinality": high_cardinality,
        "correlated_pairs": correlated_pairs,
        "skewed": skewed,
        "outliers": outliers,
        "mixed_types": mixed_types,
        "text_numbers": text_numbers,
        "issues": issues,
    }
