import pandas as pd


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Analyze basic data quality issues in a DataFrame."""

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    total_rows, total_columns = df.shape

    missing_by_column = df.isnull().sum()

    missing_values = {
        str(column): int(count)
        for column, count in missing_by_column.items()
        if count > 0
    }

    duplicate_rows = int(df.duplicated().sum())

    constant_columns = [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    high_missing_columns = [
        str(column)
        for column, count in missing_by_column.items()
        if count / total_rows >= 0.5
    ]

    column_types = {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }

    column_summary = []

    for column in df.columns:
        series = df[column]

        column_summary.append({
            "name": str(column),
            "dtype": str(series.dtype),
            "missing": int(series.isnull().sum()),
            "unique": int(series.nunique(dropna=True)),
            "missing_percentage": round(
                float(series.isnull().mean() * 100), 2
            ),
        })

    return {
        "rows": int(total_rows),
        "columns": int(total_columns),
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_values,
        "constant_columns": constant_columns,
        "high_missing_columns": high_missing_columns,
        "column_types": column_types,
        "column_summary": column_summary,
    }
