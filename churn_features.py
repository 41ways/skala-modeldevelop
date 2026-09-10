"""파이프라인 공용 전처리 정의. 학습·예측 스크립트가 함께 쓴다."""
import pandas as pd

DROP_COLS = ["CustomerId", "Surname"]
DATE_COLS = ["baseDate", "accountOpeningDate"]
NUM_COLS = [
    "CreditScore", "Age", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "TenureDays",
]
CAT_COLS = ["Geography", "Gender"]
TARGET = "Exited"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """기준일 - 계좌개설일 = 거래기간(일). 원본 날짜 컬럼과 식별자는 버린다."""
    df = df.copy()
    base = pd.to_datetime(df["baseDate"], format="mixed")
    opened = pd.to_datetime(df["accountOpeningDate"], format="mixed")
    df["TenureDays"] = (base - opened).dt.days
    return df[NUM_COLS + CAT_COLS]
