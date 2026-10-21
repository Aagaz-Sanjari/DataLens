# DataLens — Dataset Readiness Checker for Machine Learning

DataLens is a Flask-based web application that analyzes CSV datasets and helps users identify potential problems before using the data for machine learning.

It evaluates data quality, analyzes the target column, detects potential data leakage, examines feature characteristics, and generates an explainable dataset readiness score from 0 to 100.

## Project Overview

Preparing a dataset is an important step in developing reliable machine learning models. Missing values, duplicate records, imbalanced target classes, suspicious identifiers, and problematic features can make model development more difficult.

DataLens brings several preliminary dataset checks into one web application. Users can upload a CSV file, inspect analysis results, understand potential issues, and download a version of the dataset with duplicate rows removed.

**Important:** The readiness score is a rule-based heuristic. It indicates potential dataset preparation issues; it does not guarantee model accuracy or prove that a dataset is suitable for every machine learning task.

## Objectives

* Analyze the quality and structure of uploaded datasets.
* Identify missing values, duplicate rows, and constant columns.
* Infer whether a selected target is likely to represent classification or regression.
* Highlight target imbalance, rare classes, and other target-related concerns.
* Detect suspicious identifier columns and potential data leakage indicators.
* Analyze features for correlation, skewness, outliers, and high cardinality.
* Generate a readiness score with explanations and prioritized recommendations.
* Allow users to download a cleaned CSV with duplicate rows removed.

## Features

### 1. CSV Upload and Validation

* Upload CSV datasets through a web interface.
* Validate uploaded files and handle common input errors.
* Apply file-size and dataset-dimension limits.
* Preview uploaded data in the results dashboard.

### 2. Data Quality Analysis

* Count missing values by column.
* Detect duplicate rows.
* Identify constant columns.
* Flag columns with at least 50% missing values.
* Summarize column data types and missing-value percentages.

### 3. Target Column Analysis

When a target column is selected, DataLens examines its characteristics.

**Classification analysis**

* Class distribution.
* Majority-class percentage.
* Class imbalance.
* Rare classes.
* Number of distinct classes.

**Regression analysis**

* Mean and median.
* Standard deviation.
* Minimum and maximum values.
* Target skewness.

Task-type inference is heuristic, so users should verify the inferred task before relying on its results.

### 4. Potential Data Leakage Detection

* Identify suspicious identifier-like columns.
* Examine individual features for unusually strong predictive performance.
* Flag potential leakage indicators for further investigation.

A highly predictive feature is not necessarily leakage. Findings must be investigated in the context of the dataset and prediction task.

### 5. Feature Analysis

* Check the number of rows relative to the number of features.
* Identify high-cardinality categorical columns.
* Detect highly correlated numerical features.
* Flag strongly skewed numerical features.
* Identify potential numerical outliers.
* Examine mixed-type columns and numbers stored as text.

### 6. Dataset Readiness Score

DataLens generates a score between 0 and 100 using rule-based deductions.

The scoring system considers:

* Data quality.
* Target characteristics.
* Potential data leakage.
* Feature characteristics.

The dashboard displays the resulting score, grade, deduction breakdown, and prioritized recommendations.

The score is an indicative assessment based on predefined rules, not a statistically validated probability of successful model training.

### 7. Cleaned CSV Download

Users can download a version of their dataset with duplicate rows removed.

**Note:** This operation does not automatically impute missing values, encode categorical variables, scale features, or resolve every issue identified by the analysis.

## Technology Stack

| Technology   | Purpose                             |
| ------------ | ----------------------------------- |
| Python       | Application logic and data analysis |
| Flask        | Web application and routing         |
| Pandas       | Data loading and manipulation       |
| NumPy        | Numerical computations              |
| Scikit-learn | Feature-level predictive checks     |
| HTML         | Web page structure                  |
| CSS          | User interface styling              |
| JavaScript   | Client-side upload interactions     |
| Gunicorn     | Production WSGI server              |
| Render       | Planned/optional cloud deployment   |

## Project Structure

```text
DataLens/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── data_quality.py
│   ├── target_analysis.py
│   ├── leakage_detection.py
│   ├── feature_analysis.py
│   ├── readiness_score.py
│   ├── model_probe.py          # Future improvement
│   └── anomaly_detection.py    # Future improvement
│
├── templates/
│   ├── index.html
│   └── dashboard.html
│
└── static/
    └── css/
        └── style.css
```

The `model_probe.py` and `anomaly_detection.py` modules are planned improvements and are not part of the current implemented feature set unless subsequently developed.

## Installation and Setup

### Prerequisites

* Python 3.10 or a compatible Python version.
* pip package installer.
* Git (optional, for cloning the repository).

### 1. Clone the Repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DataLens
```

Alternatively, open your existing project folder in VS Code.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
Flask
pandas
numpy
scikit-learn
plotly
gunicorn
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open DataLens

Open your browser and visit:

```text
http://127.0.0.1:5000
```

Upload a CSV file, select a target column if applicable, and review the generated analysis.

## How to Use

1. Open the DataLens web application.
2. Upload a CSV dataset.
3. Select the target column if you want target-specific analysis.
4. Run the analysis.
5. Review the dataset readiness score.
6. Examine data quality and feature analysis results.
7. Review target-related findings and potential leakage indicators, when available.
8. Read the prioritized recommendations.
9. Download the dataset with duplicate rows removed if needed.

## Readiness Score Interpretation

The score is intended to summarize issues detected by predefined rules.

| Score    | Grade        |
| -------- | ------------ |
| 85–100   | Ready        |
| 70–84    | Mostly ready |
| 50–69    | Needs work   |
| Below 50 | Not ready    |

The application may label an assessment as partial when no target column is selected. Interpret the score alongside the individual findings rather than using it as the sole basis for deciding whether to train a model.

## Limitations

* The readiness score is heuristic and depends on predefined thresholds and deductions.
* Target-type inference may be incorrect for some datasets.
* Potential leakage detection uses limited feature-level checks and cannot identify every form of leakage.
* Correlation does not imply causation or automatically mean a feature should be removed.
* Outliers and skewness may be valid characteristics of a dataset.
* Removing duplicate rows does not solve all data quality problems.
* The current analysis does not guarantee that a machine learning model will perform well.
* Dataset size and analysis limits may affect which records or features are examined.

## Future Improvements

* Compare baseline machine learning models using appropriate validation.
* Add more advanced anomaly detection.
* Provide configurable data preparation recommendations.
* Expand support for larger datasets.
* Add more interactive visualizations.
* Improve task-type inference and validation strategies.
* Add automated tests for analysis modules and application routes.
* Improve deployment monitoring and error reporting.

## Deployment

The application can be deployed to a compatible hosting platform such as Render using Gunicorn.

Example production start command:

```bash
gunicorn app:app
```

Deployment should be considered complete only after the live application has been tested successfully.

**Live Demo:** Add your deployed URL here after deployment.

**GitHub Repository:** Add your repository URL here.

## Conclusion

DataLens provides an integrated starting point for evaluating CSV datasets before machine learning. By combining data quality checks, target analysis, potential leakage detection, feature analysis, and explainable readiness scoring, it helps users recognize common dataset preparation concerns before beginning model development.

The project demonstrates the use of Python, Flask, Pandas, NumPy, and Scikit-learn to build a practical data science web application.

## Author

**Aagaz Sanjari Shaikh**

Computer Science Engineering (Data Science)

Amravati, Maharashtra, India
