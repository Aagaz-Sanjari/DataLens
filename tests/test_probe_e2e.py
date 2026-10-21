import io, sys, json, re
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
import app as appmod

c = appmod.app.test_client()
rng = np.random.RandomState(0)

def post(df, **form):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    data = {"file": (buf, "data.csv"), **form}
    return c.post("/analyze", data=data, content_type="multipart/form-data")

X, y = make_classification(800, 8, n_informative=5, random_state=0)
clf = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]); clf["label"] = y

# 1. probe on
r = post(clf, target="label", probe="on"); h = r.get_data(as_text=True)
assert r.status_code == 200
for needle in ["Model Probe: Neural Network vs Random Forest", "Strong signal", "learning-curve",
               "probe-data", "plotly-basic-2.35.2", "js/dashboard.js", "Neural network (MLP)"]:
    assert needle in h, needle
payload = json.loads(re.search(r'<script id="probe-data" type="application/json">(.*?)</script>', h, re.S).group(1))
assert set(payload["curves"]) == {"Neural network (MLP)", "Random Forest"} and payload["metric"] == "ROC AUC"
print("1 probe on ........ OK  (score card still present:", "ML Readiness Score" in h, ")")

# 2. probe off
r = post(clf, target="label"); h = r.get_data(as_text=True)
assert r.status_code == 200 and "Not run" in h and "plotly" not in h
print("2 probe off ....... OK")

# 3. no target -> no probe section at all, no crash
r = post(clf); h = r.get_data(as_text=True)
assert r.status_code == 200 and "Model Probe" not in h
print("3 no target ....... OK")

# 4. regression
reg = pd.DataFrame(rng.randn(600, 5), columns=list("abcde")); reg["price"] = reg.a * 3 + reg.b + rng.randn(600) * .1
r = post(reg, target="price", probe="on"); h = r.get_data(as_text=True)
assert r.status_code == 200 and "R&sup2;" in h and "R²" in h
print("4 regression ...... OK")

# 5. tiny dataset -> message, not crash
r = post(clf.head(30), target="label", probe="on"); h = r.get_data(as_text=True)
assert r.status_code == 200 and "At least 60 are needed" in h
print("5 tiny dataset .... OK")

# 6. probe raises -> rest of report survives
orig = appmod.run_model_probe
appmod.run_model_probe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
r = post(clf, target="label", probe="on"); h = r.get_data(as_text=True)
assert r.status_code == 200 and "unexpected error" in h and "ML Readiness Score" in h
appmod.run_model_probe = orig
print("6 probe crash ..... OK  (report survives)")

# 7. hostile column name must be escaped in the new section
evil = clf.rename(columns={"f0": "<script>alert(1)</script>"})
evil["const"] = 1
r = post(evil, target="label", probe="on"); h = r.get_data(as_text=True)
assert r.status_code == 200 and "<script>alert(1)</script>" not in h
print("7 XSS column name . OK")

# 8. leak column is listed as excluded in the probe section
leak = clf.copy(); leak["leaky"] = leak["label"] * 2.0
r = post(leak, target="label", probe="on"); h = r.get_data(as_text=True)
assert "Left out of the probe" in h and "leaky (possible leakage)" in h
print("8 leak excluded ... OK")

# 9. index page renders with checkbox
h = c.get("/").get_data(as_text=True); assert 'name="probe"' in h
print("9 upload page ..... OK")
print("\nALL E2E CHECKS PASSED")
