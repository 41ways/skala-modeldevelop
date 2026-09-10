# 은행 고객 이탈 예측 실습

`생성형AI 기반 모델개발 실습.pptx` 의 Practice #1~#3 을 로컬 코드로 돌린 결과.

## 파일

| 파일 | 내용 |
| --- | --- |
| `churn_features.py` | 전처리 정의. 학습·예측이 같이 씀 |
| `01_train_rf.py` | Practice #1·#2 — RandomForest 학습, 성능·중요도, pkl 저장 |
| `02_model_compare.py` | Self-Practice Q2 — 6종 모델 비교 |
| `03_predict.py` | Practice #3 — 신규 고객 스코어링 |
| `04_prompt_strict.py` | 날짜 컬럼 처리 방식별 성능 확인 |
| `outputs/` | 리포트·성능표·스코어 CSV |

`bank_churn_model.pkl` 은 10MB라 저장소에서 뺌. `01_train_rf.py` 돌리면 다시 생김.

실행: `python 01_train_rf.py && python 02_model_compare.py && python 03_predict.py`

## 프롬프트 조건 대조

| 조건 | 구현 |
| --- | --- |
| 목표변수 Exited | `churn_features.TARGET` |
| 알고리즘 Random Forest | `RandomForestClassifier` |
| 제외 컬럼 CustomerId, Surname | `DROP_COLS` |
| Train 60 / Validation 20 / Test 20 | 층화 2단계 분할 |
| 결과 1) Train·Validation·Test 별 Accuracy·F1·AUC | Practice #1 표 |
| 결과 2) 변수 중요도 순위와 변수-Target 관계 | 변수 중요도 / 변수와 Target 의 관계 |

프롬프트에 없는 판단이 둘 있음. 결측 대체와 날짜 컬럼 처리.

날짜는 `TenureDays` 로 바꿔 썼음. 이 선택이 성능을 좌우하는지 `04_prompt_strict.py` 로 확인함.

| 처리 | Val AUC | Test AUC |
| --- | --- | --- |
| 날짜 컬럼 제외 | 0.8605 | 0.8528 |
| 계좌개설일 ordinal | 0.8714 | 0.8508 |
| TenureDays 파생 (본 구현) | 0.8713 | 0.8489 |

Test AUC 차이 0.004뿐이라 셋 다 같은 수준. `baseDate` 가 2026-02-01 단일값이라 `TenureDays` 는
계좌개설일의 선형 변환이고, 순열 중요도 기여도 0 근처. 해석 편의로 남긴 것 뿐이라고 봄.

## 전처리

- 제외: `CustomerId`, `Surname`
- 파생: `TenureDays` = `baseDate` − `accountOpeningDate` (일). 원본 날짜 컬럼은 제외
- 결측: `Geography` 42건 최빈값, `EstimatedSalary` 69건 중앙값 대체
- 범주형: `Geography`, `Gender` one-hot (`handle_unknown="ignore"`)
- 불균형: 이탈 21.3% → `class_weight="balanced_subsample"`
- 분할: Train 60% / Validation 20% / Test 20%, 층화, seed 42

대체를 파이프라인 안에 넣은 이유는 중앙값·최빈값을 Train 에서만 뽑기 위함. 전체 데이터로 먼저
채우면 Test 정보가 새어 들어감.

## Practice #1 — RandomForest

| Set | Accuracy | F1 | AUC |
| --- | --- | --- | --- |
| Train | 0.9754 | 0.9435 | 0.9966 |
| Validation | 0.8548 | 0.6258 | 0.8713 |
| Test | 0.8500 | 0.6038 | 0.8489 |

Train 과 Validation 의 AUC 차이 0.13 은 트리 계열에서 흔한 폭. Validation 과 Test 가 0.87 / 0.85 로
붙어 있어 일반화는 안정적이라고 봄.

### 변수 중요도

순열 중요도(Test AUC 하락폭) 기준: `Age` 0.138 > `NumOfProducts` 0.098 > `IsActiveMember` 0.048 >
`Balance` 0.021 > `Gender` 0.018 > `Geography` 0.012.

불순도 기준에서는 `EstimatedSalary`·`CreditScore`·`TenureDays` 가 상위에 오지만 순열 기준으로는
0 근처거나 음수. 고유값이 많은 연속형이 분기 기회를 많이 얻어 생기는 편향이고 예측력은 없다고 봄.
해석은 순열 중요도로 함.

### 변수와 Target 의 관계

- `Age` — 32세 이하 7% → 44세 초과 44.7%. 단조 증가하는 강한 신호
- `NumOfProducts` — 2개 7.5% 로 최저, 1개 28.5%, 3개 88.1%, 4개 100%. 3개 이상은 사실상 이탈 확정
- `IsActiveMember` — 비활동 29.5% vs 활동 13.6%
- `Geography` — 독일 31.1% vs 프랑스 18.3% / 스페인 17.6%
- `Gender` — 여성 25.5% vs 남성 17.7%
- `CreditScore`, `EstimatedSalary`, `HasCrCard`, `TenureDays` — 구간별 이탈률 차이 미미

## Self-Practice Q2 — 모델 비교

| Model | Val AUC | Test AUC | Test F1 |
| --- | --- | --- | --- |
| XGBoost (early stop) | 0.8759 | 0.8624 | 0.6066 |
| LightGBM (early stop) | 0.8738 | 0.8587 | 0.5943 |
| RandomForest | 0.8713 | 0.8489 | 0.6038 |
| XGBoost | 0.8635 | 0.8353 | 0.5604 |
| HistGradientBoosting | 0.8616 | 0.8387 | 0.5298 |
| LogisticRegression | 0.7772 | 0.7735 | 0.5000 |

Test AUC 최고는 early stopping 붙인 XGBoost 0.8624. 조기 종료 없이 500 트리를 그대로 돌린 부스팅은
Train AUC 가 1.0 에 붙고 Test 는 RandomForest 보다 낮게 나옴. 2,100건 규모에서 부스팅 성능은
규제와 조기 종료에 달렸다고 봄.

Accuracy 는 RandomForest 가 0.85 로 가장 높지만 이탈 21% 불균형에서 0.5 임계값이 유리하게 작용한
결과. 순위 품질을 보는 AUC 와 방향이 갈리므로, 캠페인 대상처럼 상위 N명을 뽑는 용도면 AUC 기준이 맞음.

## Practice #2 — 모델 저장

`bank_churn_model.pkl` 은 파생변수 생성 → 결측 대체 → one-hot → RandomForest 를 묶은 sklearn
Pipeline. 원본 CSV 를 그대로 넣으면 예측이 나옴.

`cloudpickle.register_pickle_by_value` 로 `churn_features` 모듈을 값으로 직렬화함. pkl 만 다른
경로로 옮겨도 `churn_features.py` 없이 로드되는 것 확인함.

## Practice #3 — 신규 고객 스코어링

`bank_churn_new.csv` 900건 → `outputs/bank_churn_new_scored.csv`

컬럼: `CustomerId`, `Surname`, `ChurnScore`, `PredictedExited`(0.5 기준), `RiskGrade`

평균 스코어 0.283, 0.5 이상 161건. 등급은 Very High 50 / High 111 / Mid 158 / Low 581.

임계값 0.5 는 기본값 뿐임. 캠페인 예산이 정해져 있으면 상위 N명을 자르는 편이 낫다고 봄.
