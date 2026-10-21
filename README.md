# DataLens — Dataset Readiness Checker for Machine Learning

**Analyze your dataset. Discover potential problems. Prepare better data for machine learning.**

DataLens is a Flask-based web application that evaluates the quality and machine learning readiness of CSV datasets. It generates a readiness score, identifies potential data issues, and provides actionable recommendations before model training.

## 🌐 Live Demo

**Try DataLens:** https://data-lens-beta.vercel.app/

## ✨ Features

### 1. Data Quality Analysis

* Detect missing values.
* Identify duplicate rows.
* Find constant columns.
* Examine column data types.
* Highlight columns with excessive missing data.

### 2. Target Variable Analysis

* Identify potential classification or regression tasks.
* Examine target distributions.
* Detect class imbalance and rare classes.
* Check for missing or constant target values.

### 3. Potential Data Leakage Detection

* Flag identifier-like columns.
* Test selected features for suspiciously high predictive performance.
* Highlight potential leakage risks for further investigation.

### 4. Feature Analysis

* Analyze feature types and cardinality.
* Identify highly correlated numerical features.
* Detect skewed numerical distributions.
* Flag potential outliers.
* Identify numerical values stored as text.

### 5. Machine Learning Readiness Score

* Generate a rule-based readiness score from 0 to 100.
* Summarize important dataset issues.
* Provide prioritized recommendations for improving data quality.

### 6. Cleaned Dataset Download

* Download a CSV with duplicate rows removed.

**Note:** The readiness score and leakage checks are diagnostic heuristics. They do not guarantee model performance or conclusively prove data leakage.

## 🛠️ Technology Stack

* **Language:** Python
* **Web Framework:** Flask
* **Data Processing:** Pandas, NumPy
* **Machine Learning Utilities:** Scikit-learn
* **Visualization:** Plotly
* **Deployment:** Vercel
* **Version Control:** Git and GitHub

## 🏗️ Project Structure

```text
DataLens/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── data_quality.py
│   ├── target_analysis.py
│   ├── leakage_detection.py
│   ├── feature_analysis.py
│   └── readiness_score.py
├── templates/
│   ├── index.html
│   └── dashboard.html
├── static/
│   └── css/
│       └── style.css
├── tests/
└── uploads/
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Aagaz-Sanjari/DataLens.git
cd DataLens
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000/
```

## 📊 How It Works

1. Upload a CSV dataset.
2. DataLens examines data quality and feature characteristics.
3. The application analyzes the target variable when applicable.
4. Potential leakage risks and other issues are flagged.
5. A readiness score and recommendations are generated.
6. Review the report and download the CSV with duplicate rows removed.

## 🎯 Project Objective

The objective of DataLens is to make preliminary dataset assessment easier for students, data analysts, and machine learning practitioners by identifying common data preparation issues before model development.

## 🔒 Limitations

* The readiness score is rule-based and is not a validated prediction of model performance.
* Leakage detection can produce false positives and false negatives.
* Automatic duplicate removal does not guarantee that every removed row is redundant for every use case.
* The application is designed for CSV datasets within its configured upload limits.
* Uploaded files should not contain sensitive information unless the deployment has been appropriately configured to handle it.

## 👨‍💻 Author

AAGAZ SANJARI
