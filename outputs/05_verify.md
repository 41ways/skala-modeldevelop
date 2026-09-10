# pkl 검증

- SHA-256: `443cc4585873c9edccbc740fbe1a7b56784c8b72f5017a43df59c281306a64a2`
- 파이프라인 단계: features → prep → model
- RandomForest: n_estimators=500, min_samples_leaf=3, random_state=42, 입력 피처 13개
- 분할 재현: Train 1260 / Validation 420 / Test 420 → `outputs/05_split_assignment.csv` 에 CustomerId 단위로 기록
- 성능표 재계산 일치: **일치**
- 같은 seed 재학습 예측 일치: **True**

## pkl 로 다시 계산한 성능

| set        |   Accuracy |     F1 |    AUC |
|:-----------|-----------:|-------:|-------:|
| Train      |     0.9754 | 0.9435 | 0.9966 |
| Validation |     0.8548 | 0.6258 | 0.8713 |
| Test       |     0.85   | 0.6038 | 0.8489 |
