"""프롬프트 문구 그대로의 변형 확인 — 날짜 파생 없이 CustomerId·Surname 만 제외."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
raw = pd.read_csv("bank_churn_train.csv")
y = raw["Exited"]
X = raw.drop(columns=["CustomerId", "Surname", "Exited"])

variants = {
    "A. 날짜 컬럼 제외 (프롬프트 최소 해석)": X.drop(columns=["baseDate", "accountOpeningDate"]),
    "B. 계좌개설일 ordinal": X.drop(columns=["baseDate"]).assign(
        accountOpeningDate=pd.to_datetime(X["accountOpeningDate"], format="mixed").map(pd.Timestamp.toordinal)),
}
rows = []
for name, Xv in variants.items():
    num = [c for c in Xv.columns if Xv[c].dtype.kind in "if"]
    cat = [c for c in Xv.columns if c not in num]
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                              ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                         class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
    ])
    Xtr, Xtmp, ytr, ytmp = train_test_split(Xv, y, test_size=0.4, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=SEED)
    pipe.fit(Xtr, ytr)
    r = {"variant": name}
    for tag, Xs, ys in [("Train", Xtr, ytr), ("Val", Xva, yva), ("Test", Xte, yte)]:
        p, pr = pipe.predict(Xs), pipe.predict_proba(Xs)[:, 1]
        r[f"{tag}_Acc"] = accuracy_score(ys, p); r[f"{tag}_F1"] = f1_score(ys, p); r[f"{tag}_AUC"] = roc_auc_score(ys, pr)
    rows.append(r)

res = pd.DataFrame(rows)
res.round(4).to_csv("outputs/04_prompt_strict.csv", index=False)
print(res.round(4).to_string(index=False))
