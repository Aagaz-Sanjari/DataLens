import io
import logging
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from src.data_quality import analyze_data_quality
from src.target_analysis import analyze_target
from src.leakage_detection import detect_leakage
from src.feature_analysis import analyze_features
from src.readiness_score import compute_readiness

app = Flask(__name__)

# Maximum uploaded file size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".csv"}
MAX_COLUMNS = 200
MAX_ROWS = 200_000

logging.basicConfig(level=logging.INFO)


def allowed_file(filename):
    """Check whether the uploaded file is a CSV."""
    return (
        bool(filename)
        and Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
    )


def read_csv_file(file_storage):
    """Read and validate an uploaded CSV file."""
    df = pd.read_csv(file_storage)

    if df.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    if len(df.columns) == 0:
        raise ValueError("The CSV must contain at least one column.")

    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(
            f"This version supports a maximum of {MAX_COLUMNS} columns."
        )

    if len(df) > MAX_ROWS:
        raise ValueError(
            f"This version supports a maximum of {MAX_ROWS:,} rows."
        )

    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names were found. "
            "Please rename them and upload the file again."
        )

    return df


@app.route("/", methods=["GET"])
def index():
    """Display the DataLens upload page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze the uploaded dataset and display the results."""
    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:
        return render_template(
            "index.html",
            error="Please select a CSV file before analyzing."
        ), 400

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "index.html",
            error="Invalid file type. Please upload a CSV file."
        ), 400

    try:
        df = read_csv_file(uploaded_file)

        # Run the data quality analysis.
        quality = analyze_data_quality(df)

        # Optional: target and leakage checks when a target column is chosen.
        target = (request.form.get("target") or "").strip()
        target_report = None
        leakage_report = None

        if target:
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' was not found.")

            target_report = analyze_target(df, target)
            leakage_report = detect_leakage(
                df, target, target_report["task_type"]
            )

        feature_report = analyze_features(df, target or None)
        readiness = compute_readiness(
            quality, target_report, leakage_report, feature_report
        )

        # Generate a safe HTML preview.
        preview = df.head(10).to_html(
            index=False,
            classes="data-table",
            border=0,
            na_rep="Missing",
            escape=True
        )

        return render_template(
            "dashboard.html",
            filename=Path(uploaded_file.filename).name,
            quality=quality,
            target_report=target_report,
            leakage_report=leakage_report,
            feature_report=feature_report,
            readiness=readiness,
            preview=preview,
            columns=df.columns.tolist()
        )

    except (
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError
    ) as exc:
        return render_template(
            "index.html",
            error=f"Could not analyze this CSV: {exc}"
        ), 400

    except Exception:
        app.logger.exception("Dataset analysis failed")

        return render_template(
            "index.html",
            error="An unexpected error occurred. Please try another CSV file."
        ), 500


@app.route("/download-cleaned", methods=["POST"])
def download_cleaned():
    """Download a CSV after removing duplicate rows."""
    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:
        return "Please select a CSV file.", 400

    if not allowed_file(uploaded_file.filename):
        return "Only CSV files are supported.", 400

    try:
        df = read_csv_file(uploaded_file)

        # Remove duplicate rows.
        cleaned_df = df.drop_duplicates()

        output = io.BytesIO(
            cleaned_df.to_csv(index=False).encode("utf-8")
        )
        output.seek(0)

        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="datalens_cleaned.csv"
        )

    except (
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError
    ) as exc:
        return f"Could not process the CSV: {exc}", 400

    except Exception:
        app.logger.exception("CSV cleaning failed")
        return "An unexpected error occurred while cleaning the CSV.", 500


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    """Handle files that exceed the upload limit."""
    return render_template(
        "index.html",
        error="File too large. The maximum upload size is 10 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=False)
