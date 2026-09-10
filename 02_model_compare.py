"""Self-Practice Q2 — RF 외 모델 비교. Validation AUC로 고르고 Test AUC로 확인."""
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import churn_features as cf

SEED = 42
raw = pd.read_csv("bank_churn_train.csv")
X, y = raw.drop(columns=[cf.TARGET]), raw[cf.TARGET]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()


def prep(scale=False):
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), cf.NUM_COLS),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cf.CAT_COLS),
    ])


models = {
    "LogisticRegression": (LogisticRegression(max_iter=2000, class_weight="balanced"), True),
    "RandomForest": (RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), False),
    "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                            max_leaf_nodes=15, random_state=SEED), False),
    "XGBoost": (XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=pos_w,
                              eval_metric="auc", n_jobs=-1, random_state=SEED), False),
    "LightGBM": (lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15,
                                    min_child_samples=20, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, class_weight="balanced",
                                    n_jobs=-1, random_state=SEED, verbose=-1), False),
}

rows = []
for name, (clf, scale) in models.items():
    pipe = Pipeline([("features", FunctionTransformer(cf.add_features)), ("prep", prep(scale)), ("model", clf)])
    pipe.fit(X_tr, y_tr)
    r = {"model": name}
    for tag, Xs, ys in [("Train", X_tr, y_tr), ("Val", X_va, y_va), ("Test", X_te, y_te)]:
        p, pr = pipe.predict(Xs), pipe.predict_proba(Xs)[:, 1]
        r[f"{tag}_AUC"] = roc_auc_score(ys, pr)
        r[f"{tag}_F1"] = f1_score(ys, p)
        r[f"{tag}_Acc"] = accuracy_score(ys, p)
    rows.append(r)

res = pd.DataFrame(rows).sort_values("Val_AUC", ascending=False)
cols = ["model", "Train_AUC", "Val_AUC", "Test_AUC", "Test_F1", "Test_Acc"]
res[cols].round(4).to_csv("outputs/02_model_compare.csv", index=False)
best_val = res.iloc[0]["model"]
best_test = res.sort_values("Test_AUC", ascending=False).iloc[0]["model"]
with open("outputs/02_model_compare.md", "w") as f:
    f.write("# Self-Practice Q2 — 모델 비교\n\n")
    f.write(res[cols].round(4).to_markdown(index=False))
    f.write(f"\n\n- Validation AUC 최고: **{best_val}**\n- Test AUC 최고: **{best_test}**\n")
print(res[cols].round(4).to_string(index=False))
print("val-best:", best_val, "| test-best:", best_test)

# --- 부스팅 모델 과적합 보정: Validation 으로 early stopping ---
pre = prep()
F_tr, F_va, F_te = (cf.add_features(d) for d in (X_tr, X_va, X_te))
Xt_tr = pre.fit_transform(F_tr, y_tr)
Xt_va, Xt_te = pre.transform(F_va), pre.transform(F_te)

es_rows = []
xgb_es = XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=3, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=5,
                       scale_pos_weight=pos_w, eval_metric="auc", early_stopping_rounds=50,
                       n_jobs=-1, random_state=SEED)
xgb_es.fit(Xt_tr, y_tr, eval_set=[(Xt_va, y_va)], verbose=False)

lgb_es = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                            class_weight="balanced", n_jobs=-1, random_state=SEED, verbose=-1)
lgb_es.fit(Xt_tr, y_tr, eval_set=[(Xt_va, y_va)], eval_metric="auc",
           callbacks=[lgb.early_stopping(50, verbose=False)])

for name, m in [("XGBoost (early stop)", xgb_es), ("LightGBM (early stop)", lgb_es)]:
    r = {"model": name}
    for tag, Xs, ys in [("Train", Xt_tr, y_tr), ("Val", Xt_va, y_va), ("Test", Xt_te, y_te)]:
        p, pr = m.predict(Xs), m.predict_proba(Xs)[:, 1]
        r[f"{tag}_AUC"] = roc_auc_score(ys, pr); r[f"{tag}_F1"] = f1_score(ys, p); r[f"{tag}_Acc"] = accuracy_score(ys, p)
    es_rows.append(r)

final = pd.concat([res, pd.DataFrame(es_rows)]).sort_values("Val_AUC", ascending=False)
final[cols].round(4).to_csv("outputs/02_model_compare.csv", index=False)
best_val = final.iloc[0]["model"]
best_test = final.sort_values("Test_AUC", ascending=False).iloc[0]["model"]
with open("outputs/02_model_compare.md", "w") as f:
    f.write("# Self-Practice Q2 — 모델 비교\n\n")
    f.write(final[cols].round(4).to_markdown(index=False))
    f.write(f"\n\n- Validation AUC 최고: **{best_val}**\n- Test AUC 최고: **{best_test}**\n")
print("\n[early stopping 포함]")
print(final[cols].round(4).to_string(index=False))
print("val-best:", best_val, "| test-best:", best_test)
