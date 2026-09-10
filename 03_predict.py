"""Practice #3 — 저장된 파이프라인으로 신규 고객 이탈 Score 생성."""
import pickle

import pandas as pd

with open("bank_churn_model.pkl", "rb") as f:
    pipe = pickle.load(f)

new = pd.read_csv("bank_churn_new.csv")
score = pipe.predict_proba(new)[:, 1]

out = pd.DataFrame({
    "CustomerId": new["CustomerId"],
    "Surname": new["Surname"],
    "ChurnScore": score.round(4),
    "PredictedExited": (score >= 0.5).astype(int),
    "RiskGrade": pd.cut(score, [-0.01, 0.3, 0.5, 0.7, 1.0], labels=["Low", "Mid", "High", "Very High"]),
}).sort_values("ChurnScore", ascending=False)

out.to_csv("outputs/bank_churn_new_scored.csv", index=False)
print(f"{len(out)}건 스코어링, 평균 {score.mean():.3f}, 0.5 이상 {int((score >= 0.5).sum())}건")
print(out["RiskGrade"].value_counts().reindex(["Very High", "High", "Mid", "Low"]).to_string())
print("\n상위 5명:")
print(out.head(5).to_string(index=False))
