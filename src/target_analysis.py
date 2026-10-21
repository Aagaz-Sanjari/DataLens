import numpy as np
import pandas as pd

RARE_CLASS_MIN_COUNT = 10
IMBALANCE_WARN_RATIO = 3.0
IMBALANCE_SEVERE_RATIO = 10.0


def infer_task_type(series: pd.Series) -> str:
    """Guess whether a target column is a classification or regression target."""
    clean = series.dropna()
    n_unique = clean.nunique()

    if (
        pd.api.types.is_bool_dtype(clean)
        or pd.api.types.is_object_dtype(clean)
        or isinstance(clean.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(clean)
    ):
        return "classification"

    if not pd.api.types.is_numeric_dtype(clean):
        return "classification"

    # Numeric targets: few distinct whole numbers behave like class labels.
    is_whole = bool(np.all(np.isclose(clean % 1, 0)))
    if is_whole and (n_unique <= 10 or n_unique / max(len(clean), 1) < 0.02):
        return "classification"

    return "regression"


def analyze_target(df: pd.DataFrame, target: str) -> dict:
    """Check the target column for problems that hurt model training."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    series = df[target]
    total = len(series)
    missing = int(series.isnull().sum())
    clean = series.dropna()

    if clean.empty:
        raise ValueError(f"Target column '{target}' contains only missing values.")

    task_type = infer_task_type(series)
    n_unique = int(clean.nunique())
    issues = []

    if missing > 0:
        pct = missing / total * 100
        issues.append({
            "severity": "high" if pct >= 5 else "medium",
            "message": f"{missing:,} rows ({pct:.1f}%) have no target value "
                       "and cannot be used for training.",
        })

    if n_unique <= 1:
        issues.append({
            "severity": "high",
            "message": "The target has a single value, so there is "
                       "nothing for a model to learn.",
        })

    report = {
        "target": str(target),
        "task_type": task_type,
        "missing": missing,
        "unique_values": n_unique,
        "distribution": [],
        "stats": None,
        "imbalance_ratio": None,
        "majority_percentage": None,
        "issues": issues,
    }

    if task_type == "classification":
        counts = clean.value_counts()
        shown = counts.head(15)
        report["distribution"] = [
            {
                "label": str(label),
                "count": int(count),
                "percentage": round(count / len(clean) * 100, 2),
            }
            for label, count in shown.items()
        ]

        report["majority_percentage"] = round(counts.iloc[0] / len(clean) * 100, 2)

        if len(counts) >= 2:
            ratio = float(counts.iloc[0] / counts.iloc[-1])
            report["imbalance_ratio"] = round(ratio, 2)

            if ratio >= IMBALANCE_SEVERE_RATIO:
                issues.append({
                    "severity": "high",
                    "message": f"Severe class imbalance (largest class is "
                               f"{ratio:.1f}x the smallest). Accuracy will be "
                               "misleading; use recall, F1 or class weights.",
                })
            elif ratio >= IMBALANCE_WARN_RATIO:
                issues.append({
                    "severity": "medium",
                    "message": f"Moderate class imbalance ({ratio:.1f}x). "
                               "Consider stratified splits and class weights.",
                })

        rare = counts[counts < RARE_CLASS_MIN_COUNT]
        if len(rare) > 0 and len(counts) > 1:
            issues.append({
                "severity": "high",
                "message": f"{len(rare)} class(es) have fewer than "
                           f"{RARE_CLASS_MIN_COUNT} examples, which is too few "
                           "to learn from or to validate on.",
            })

        if len(counts) > 20:
            issues.append({
                "severity": "medium",
                "message": f"{len(counts)} distinct classes. Many-class "
                           "problems need far more data per class. If this "
                           "target is really continuous, treat it as regression.",
            })

    else:
        values = pd.to_numeric(clean, errors="coerce").dropna()
        skew = float(values.skew()) if len(values) > 2 else 0.0
        report["stats"] = {
            "mean": round(float(values.mean()), 4),
            "median": round(float(values.median()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "skew": round(skew, 3),
        }

        if abs(skew) >= 1:
            issues.append({
                "severity": "medium",
                "message": f"The target is strongly skewed (skew = {skew:.2f}). "
                           "A log or power transform may improve results.",
            })

    return report
