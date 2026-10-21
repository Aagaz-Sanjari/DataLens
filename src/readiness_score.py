"""Transparent, rule-based ML readiness score.

The score starts at 100. Every rule that fires subtracts a fixed number of
points and records WHY, so the dashboard can show exactly where each point
went. Each category has a cap so no single area consumes the whole score.

This is a heuristic for quick screening. It is not a validated predictor of
model performance.
"""

CATEGORY_CAPS = {
    "Data quality": 35,
    "Target": 45,
    "Leakage": 35,
    "Features": 30,
}


def _grade(score: int):
    if score >= 85:
        return "Ready", "badge-success"
    if score >= 70:
        return "Mostly ready", "badge-info"
    if score >= 50:
        return "Needs work", "badge-warning"
    return "Not ready", "badge-danger"


def compute_readiness(quality, target_report=None, leakage_report=None, feature_report=None) -> dict:
    deductions = []

    def add(category, rule, points, reason):
        deductions.append({"category": category, "rule": rule,
                           "points": points, "reason": reason})

    rows = max(quality["rows"], 1)
    columns = max(quality["columns"], 1)

    # ---------------- Data quality ----------------
    dup_pct = quality["duplicate_rows"] / rows * 100
    if dup_pct >= 10:
        add("Data quality", "Duplicate rows", 8, f"{dup_pct:.1f}% of rows are exact duplicates.")
    elif dup_pct >= 1:
        add("Data quality", "Duplicate rows", 4, f"{dup_pct:.1f}% of rows are exact duplicates.")
    elif quality["duplicate_rows"] > 0:
        add("Data quality", "Duplicate rows", 1, f"{quality['duplicate_rows']} duplicate row(s).")

    missing_cells = sum(quality["missing_values"].values())
    missing_share = missing_cells / (rows * columns) * 100
    if missing_share >= 20:
        add("Data quality", "Missing values", 12, f"{missing_share:.1f}% of all cells are empty.")
    elif missing_share >= 5:
        add("Data quality", "Missing values", 6, f"{missing_share:.1f}% of all cells are empty.")
    elif missing_cells > 0:
        add("Data quality", "Missing values", 2, f"{missing_share:.1f}% of all cells are empty.")

    for column in quality["high_missing_columns"][:3]:
        add("Data quality", "Mostly-empty column", 4, f"'{column}' is at least 50% empty.")

    for column in quality["constant_columns"][:3]:
        add("Data quality", "Constant column", 2, f"'{column}' has one value everywhere and carries no signal.")

    # ---------------- Target ----------------
    if target_report:
        if target_report["unique_values"] <= 1:
            add("Target", "Single-valued target", 40, "The target never changes, so nothing can be learned.")

        if target_report["missing"] > 0:
            pct = target_report["missing"] / rows * 100
            add("Target", "Missing target", 8 if pct >= 5 else 3,
                f"{target_report['missing']:,} rows ({pct:.1f}%) have no target value.")

        ratio = target_report.get("imbalance_ratio")
        if ratio is not None:
            if ratio >= 10:
                add("Target", "Severe class imbalance", 12, f"Largest class is {ratio:.1f}x the smallest.")
            elif ratio >= 3:
                add("Target", "Class imbalance", 5, f"Largest class is {ratio:.1f}x the smallest.")

        for issue in target_report["issues"]:
            if "fewer than" in issue["message"]:
                add("Target", "Rare classes", 10, issue["message"])
            elif "strongly skewed" in issue["message"]:
                add("Target", "Skewed target", 3, issue["message"])
            elif "distinct classes" in issue["message"]:
                add("Target", "Many classes", 4, issue["message"])

    # ---------------- Leakage ----------------
    if leakage_report:
        for finding in leakage_report["findings"]:
            if finding["severity"] == "high":
                add("Leakage", "Possible leakage", 15, f"'{finding['column']}' predicts the target almost perfectly on its own.")
            else:
                add("Leakage", "Identifier column", 3, f"'{finding['column']}' looks like a row identifier.")

    # ---------------- Features ----------------
    if feature_report:
        n_rows = feature_report["rows"]
        if n_rows < 100:
            add("Features", "Very few rows", 20, f"Only {n_rows} rows.")
        elif n_rows < 500:
            add("Features", "Few rows", 10, f"Only {n_rows} rows.")

        if feature_report["rows_per_feature"] < 10:
            add("Features", "Rows per feature", 8, f"Just {feature_report['rows_per_feature']} rows per feature.")

        for item in feature_report["mixed_types"]:
            add("Features", "Mixed-type column", 3, f"'{item['column']}' mixes numbers and text.")
        for item in feature_report["text_numbers"]:
            add("Features", "Numbers stored as text", 2, f"'{item['column']}' should be numeric.")
        for item in feature_report["high_cardinality"]:
            add("Features", "High-cardinality categorical", 2, f"'{item['column']}' has {item['unique']} categories.")
        for pair in feature_report["correlated_pairs"]:
            add("Features", "Redundant features", 2, f"'{pair['a']}' and '{pair['b']}' have |r| = {pair['correlation']}.")
        for item in feature_report["skewed"][:5]:
            add("Features", "Skewed feature", 1, f"'{item['column']}' skew = {item['skew']}.")
        for item in feature_report["outliers"][:5]:
            add("Features", "Many outliers", 1, f"'{item['column']}' has {item['percentage']}% outliers.")

    # ---------------- Apply category caps ----------------
    totals = {}
    for d in deductions:
        totals[d["category"]] = totals.get(d["category"], 0) + d["points"]

    applied = {c: min(t, CATEGORY_CAPS[c]) for c, t in totals.items()}
    total_deducted = sum(applied.values())
    score = max(0, 100 - total_deducted)
    grade, badge = _grade(score)

    complete = target_report is not None
    if not complete:
        # Without a target, leakage and target checks never ran, so don't claim "Ready".
        grade = f"{grade} (partial check)"
        badge = "badge-neutral"

    ordered = sorted(deductions, key=lambda d: d["points"], reverse=True)

    return {
        "score": int(score),
        "grade": grade,
        "badge": badge,
        "deductions": ordered,
        "category_totals": [
            {"category": c, "points": applied[c], "capped": totals[c] > CATEGORY_CAPS[c]}
            for c in applied
        ],
        "fix_first": ordered[:3],
        "complete": complete,
    }
