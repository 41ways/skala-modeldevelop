# Self-Practice Q2 — 모델 비교

| model                 |   Train_AUC |   Val_AUC |   Test_AUC |   Test_F1 |   Test_Acc |
|:----------------------|------------:|----------:|-----------:|----------:|-----------:|
| XGBoost (early stop)  |      0.9221 |    0.8759 |     0.8624 |    0.6066 |     0.8024 |
| LightGBM (early stop) |      0.937  |    0.8738 |     0.8587 |    0.5943 |     0.7952 |
| RandomForest          |      0.9966 |    0.8713 |     0.8489 |    0.6038 |     0.85   |
| XGBoost               |      0.9998 |    0.8635 |     0.8353 |    0.5604 |     0.8095 |
| HistGradientBoosting  |      1      |    0.8616 |     0.8387 |    0.5298 |     0.831  |
| LightGBM              |      1      |    0.8589 |     0.8078 |    0.5238 |     0.8095 |
| LogisticRegression    |      0.7699 |    0.7772 |     0.7735 |    0.5    |     0.7095 |

- Validation AUC 최고: **XGBoost (early stop)**
- Test AUC 최고: **XGBoost (early stop)**
