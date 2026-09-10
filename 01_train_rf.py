"""Practice #1 + #2 — RandomForest 학습, 성능 비교, 변수 중요도, 파이프라인 저장."""
import json

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

import churn_features as cf

cloudpickle.register_pickle_by_value(cf)  # pkl 하나만으로 동작하도록 함수를 값으로 직렬화

SEED = 42
raw = pd.read_csv("bank_churn_train.csv")
X, y = raw.drop(columns=[cf.TARGET]), raw[cf.TARGET]

# 60 / 20 / 20 층화 분할
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), cf.NUM_COLS),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), cf.CAT_COLS),
])

pipe = Pipeline([
    ("features", FunctionTransformer(cf.add_features)),
    ("prep", pre),
    ("model", RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
])
pipe.fit(X_tr, y_tr)


def scores(name, Xs, ys):
    p = pipe.predict(Xs)
    pr = pipe.predict_proba(Xs)[:, 1]
    return {"set": name, "n": len(ys), "Accuracy": accuracy_score(ys, p),
            "F1": f1_score(ys, p), "AUC": roc_auc_score(ys, pr)}


perf = pd.DataFrame([scores("Train", X_tr, y_tr), scores("Validation", X_va, y_va), scores("Test", X_te, y_te)])

# 변수 중요도 — 불순도 기반 + 순열 기반(Test)
names = pipe.named_steps["prep"].get_feature_names_out()
names = [n.split("__", 1)[1] for n in names]
imp = pd.DataFrame({"feature": names, "impurity": pipe.named_steps["model"].feature_importances_})
perm = permutation_importance(pipe, X_te, y_te, n_repeats=20, random_state=SEED,
                              scoring="roc_auc", n_jobs=-1)
perm_df = pd.DataFrame({"feature": X.columns, "perm_auc_drop": perm.importances_mean}).sort_values(
    "perm_auc_drop", ascending=False)
imp = imp.sort_values("impurity", ascending=False)

# 변수-타깃 관계
feat = cf.add_features(raw).assign(Exited=y)
rel = []
for col in ["Age", "Balance", "CreditScore", "EstimatedSalary", "TenureDays"]:
    q = pd.qcut(feat[col], 4, duplicates="drop")
    g = feat.groupby(q, observed=True)["Exited"].agg(["mean", "size"])
    rel.append((col, g))
for col in ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard"]:
    g = feat.groupby(col, dropna=False, observed=True)["Exited"].agg(["mean", "size"])
    rel.append((col, g))

with open("outputs/01_rf_report.md", "w") as f:
    f.write("# Practice #1 — RandomForest 결과\n\n## 성능 (Train/Validation/Test)\n\n")
    f.write(perf.round(4).to_markdown(index=False))
    f.write("\n\n## 변수 중요도 — 불순도 기반\n\n")
    f.write(imp.round(4).to_markdown(index=False))
    f.write("\n\n## 변수 중요도 — 순열 기반 (Test AUC 하락폭)\n\n")
    f.write(perm_df.round(4).to_markdown(index=False))
    f.write("\n\n## 변수별 이탈률\n")
    for col, g in rel:
        g = g.rename(columns={"mean": "이탈률", "size": "건수"})
        f.write(f"\n### {col}\n\n" + g.round(3).to_markdown() + "\n")

with open("bank_churn_model.pkl", "wb") as f:
    cloudpickle.dump(pipe, f)

perf.round(4).to_csv("outputs/01_rf_performance.csv", index=False)
print(perf.round(4).to_string(index=False))
print("\nTop5 impurity:", ", ".join(imp.head(5)["feature"]))
print("Top5 perm    :", ", ".join(perm_df.head(5)["feature"]))
