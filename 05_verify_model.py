"""저장된 pkl 이 보고한 성능표를 낸 그 모델이 맞는지 검증."""
import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import churn_features as cf

SEED = 42
raw = pd.read_csv("bank_churn_train.csv")
X, y = raw.drop(columns=[cf.TARGET]), raw[cf.TARGET]

# 1) 분할 재현 — 코드와 seed 가 같으면 같은 행이 나온다
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

split = pd.concat([
    pd.DataFrame({"CustomerId": X_tr["CustomerId"], "split": "Train", "Exited": y_tr}),
    pd.DataFrame({"CustomerId": X_va["CustomerId"], "split": "Validation", "Exited": y_va}),
    pd.DataFrame({"CustomerId": X_te["CustomerId"], "split": "Test", "Exited": y_te}),
]).sort_values(["split", "CustomerId"])
split.to_csv("outputs/05_split_assignment.csv", index=False)

# 2) pkl 지문
digest = hashlib.sha256(open("bank_churn_model.pkl", "rb").read()).hexdigest()
with open("bank_churn_model.pkl", "rb") as f:
    pipe = pickle.load(f)

# 3) pkl 로 성능표 재계산 → 보고값과 대조
rows = []
for tag, Xs, ys in [("Train", X_tr, y_tr), ("Validation", X_va, y_va), ("Test", X_te, y_te)]:
    p, pr = pipe.predict(Xs), pipe.predict_proba(Xs)[:, 1]
    rows.append({"set": tag, "Accuracy": accuracy_score(ys, p), "F1": f1_score(ys, p), "AUC": roc_auc_score(ys, pr)})
recomputed = pd.DataFrame(rows).round(4)
reported = pd.read_csv("outputs/01_rf_performance.csv")[["set", "Accuracy", "F1", "AUC"]].round(4)
match = recomputed.equals(reported)

# 4) 같은 코드·같은 seed 로 다시 학습해서 pkl 과 예측이 일치하는지
fresh_ok = None
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
    fresh = Pipeline([
        ("features", FunctionTransformer(cf.add_features)),
        ("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), cf.NUM_COLS),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cf.CAT_COLS)])),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                         class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
    ]).fit(X_tr, y_tr)
    new = pd.read_csv("bank_churn_new.csv")
    fresh_ok = bool(np.allclose(fresh.predict_proba(new)[:, 1], pipe.predict_proba(new)[:, 1], atol=0))
except Exception as e:  # noqa: BLE001
    fresh_ok = f"확인 실패: {e}"

rf = pipe.named_steps["model"]
lines = [
    "# pkl 검증",
    "",
    f"- SHA-256: `{digest}`",
    f"- 파이프라인 단계: {' → '.join(n for n, _ in pipe.steps)}",
    f"- RandomForest: n_estimators={rf.n_estimators}, min_samples_leaf={rf.min_samples_leaf}, "
    f"random_state={rf.random_state}, 입력 피처 {rf.n_features_in_}개",
    f"- 분할 재현: Train {len(X_tr)} / Validation {len(X_va)} / Test {len(X_te)} "
    f"→ `outputs/05_split_assignment.csv` 에 CustomerId 단위로 기록",
    f"- 성능표 재계산 일치: **{'일치' if match else '불일치'}**",
    f"- 같은 seed 재학습 예측 일치: **{fresh_ok}**",
    "",
    "## pkl 로 다시 계산한 성능",
    "",
    recomputed.to_markdown(index=False),
]
open("outputs/05_verify.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:8]))
print(recomputed.to_string(index=False))
