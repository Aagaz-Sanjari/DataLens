import sys, time, warnings
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.datasets import make_classification, make_regression
from src.model_probe import run_model_probe
from src.target_analysis import analyze_target
from src.leakage_detection import detect_leakage

rng = np.random.RandomState(0)

def show(label, r):
    print(f"\n=== {label} ===")
    if not r["ran"]:
        print("DID NOT RUN:", r["reason"]); return
    print(f"{r['metric']}  baseline={r['baseline_score']}  rows={r['rows_used']} feats={r['features_used']}  {r['seconds']}s")
    for m in r["models"]:
        print(f"  {m['name']:<22} val={m['score']:.3f}±{m['std']:.3f} train={m['train_score']:.3f} skill={m['skill']:.2f}  fit={m['fit_seconds']}s")
    print("  curve points:", {k: len(v) for k, v in r["curves"].items()})
    print("  findings:", [f["title"] for f in r["findings"]])
    if r["excluded"]: print("  excluded:", r["excluded"])
    if r["notes"]: print("  notes:", r["notes"])

def run(df, target, **kw):
    t = analyze_target(df, target)
    l = detect_leakage(df, target, t["task_type"])
    return run_model_probe(df, target, t["task_type"], l, **kw), t["task_type"]

# 1. strong signal classification
X, y = make_classification(1500, 10, n_informative=6, random_state=0, class_sep=1.5)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]); df["y"] = y
r, tt = run(df, "y"); show("1 signal classification", r); assert tt == "classification" and r["best_skill"] > 0.5

# 2. pure noise classification
df = pd.DataFrame(rng.randn(1500, 10), columns=[f"f{i}" for i in range(10)]); df["y"] = rng.randint(0, 2, 1500)
r, _ = run(df, "y"); show("2 noise classification", r); assert r["best_skill"] < 0.15, r["best_skill"]

# 3. regression with signal
X, y = make_regression(1200, 8, n_informative=5, noise=10, random_state=0)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]); df["price"] = y * 1000 + 50000
r, tt = run(df, "price"); show("3 regression signal", r); assert tt == "regression" and r["best_skill"] > 0.7

# 4. regression noise
df = pd.DataFrame(rng.randn(1000, 6), columns=[f"f{i}" for i in range(6)]); df["price"] = rng.randn(1000) * 100
r, _ = run(df, "price"); show("4 regression noise", r); assert r["best_skill"] < 0.1

# 5. imbalanced + perfect leak: leak column must be excluded, not inflate the score
n = 2000
y = (rng.rand(n) < 0.05).astype(int)
df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n), "leak": y * 3.0 + 1, "row_id": np.arange(n)})
df["y"] = y
r, _ = run(df, "y"); show("5 imbalanced + leak + id", r)
assert {e["column"] for e in r["excluded"]} >= {"leak", "row_id"}; assert r["best_skill"] < 0.2

# 6. messy mixed types: categoricals, bools, NaN, text numbers, dates-as-strings, high cardinality
n = 800
df = pd.DataFrame({
    "cat": rng.choice(list("abc"), n),
    "flag": rng.choice([True, False], n),
    "text_num": rng.randn(n).round(2).astype(str),
    "maybe_nan": np.where(rng.rand(n) < .2, np.nan, rng.randn(n)),
    "highcard": [f"c{i}" for i in rng.randint(0, 200, n)],
    "date": pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
    "mostly_empty": np.where(rng.rand(n) < .8, np.nan, 1.0),
    "const": 1,
})
df.loc[rng.rand(n) < .1, "cat"] = np.nan
df["y"] = np.where(df["cat"] == "a", "yes", "no")
r, _ = run(df, "y"); show("6 messy mixed types", r); assert r["ran"]

# 7. tiny dataset
df = pd.DataFrame({"a": rng.randn(30), "y": rng.randint(0, 2, 30)})
r, _ = run(df, "y"); show("7 tiny", r); assert not r["ran"]

# 8. rare class dropped
n = 600
df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
df["y"] = rng.choice(["x", "y"], n); df.loc[:1, "y"] = "rare"
r, _ = run(df, "y"); show("8 rare class", r); assert r["ran"] and r["notes"]

# 9. multiclass
X, y = make_classification(1500, 12, n_informative=8, n_classes=4, random_state=1)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]); df["y"] = y
r, _ = run(df, "y"); show("9 multiclass", r); assert abs(r["baseline_score"] - 0.5) < 0.01

# 10. large -> sampling + timing, and tight budget truncation
X, y = make_classification(20000, 20, n_informative=10, random_state=2)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]); df["y"] = y
r, _ = run(df, "y"); show("10 large (sampled)", r); assert r["sampled"] and r["rows_used"] <= 3100
r, _ = run(df, "y", time_budget=1.0); show("10b tight budget", r)
assert r["ran"] and any("time budget" in n for n in r["notes"])
print("\nALL ASSERTIONS PASSED")
