"""데이터 전처리 함수"""
import numpy as np
import pandas as pd
from typing import Tuple, List
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, MISSING_THRESHOLD


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM 호환을 위해 컬럼명에서 JSON 특수 문자 제거"""
    import re
    rename_map = {}
    for col in df.columns:
        new_col = re.sub(r'[{}\[\]"\\,()=]', '_', col)
        # 연속 밑줄 제거
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        if new_col != col:
            rename_map[col] = new_col
    if rename_map:
        df = df.rename(columns=rename_map)
        tqdm.write(f"  컬럼명 정리: {len(rename_map)}개 컬럼 이름 변경")
    return df


def remove_high_missing_columns(
    df: pd.DataFrame, threshold: float = 0.7, preserve_cols: List[str] = None
) -> pd.DataFrame:
    """결측 비율이 높은 컬럼 제거"""
    if preserve_cols is None:
        preserve_cols = []

    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio >= threshold].index.tolist()
    cols_to_drop = [col for col in cols_to_drop if col not in preserve_cols]

    if cols_to_drop:
        tqdm.write(f"  결측 비율 {threshold*100:.0f}% 이상 컬럼 제거: {len(cols_to_drop)}개")
        preserved = [col for col in preserve_cols if col in df.columns]
        if preserved:
            tqdm.write(f"  보존된 컬럼: {preserved}")
        df = df.drop(columns=cols_to_drop)
    return df


def handle_missing_values(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """결측값 처리"""
    df = df.copy()

    # 범주형 컬럼 결측값을 "미상"으로 대체
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("미상")

    # 수치형 컬럼 결측값을 중앙값으로 대체
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_with_na = [col for col in numeric_cols if df[col].isnull().any()]
    for col in tqdm(cols_with_na, desc="  수치형 결측 대체", unit="col", leave=False):
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    return df


def clip_outliers(
    df: pd.DataFrame, columns: List[str], method: str = "iqr", factor: float = 1.5
) -> pd.DataFrame:
    """이상치 클리핑"""
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < factor]

    return df


def log_transform_target(y: pd.Series) -> pd.Series:
    """target 로그 변환 (log1p)"""
    return np.log1p(y)


def inverse_log_transform(y: np.ndarray) -> np.ndarray:
    """로그 변환 역변환 (expm1)"""
    return np.expm1(y)


def identify_categorical_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """범주형 컬럼 식별"""
    if exclude_cols is None:
        exclude_cols = []

    categorical_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == "object":
            categorical_cols.append(col)
        elif df[col].dtype in ["int64", "float64"]:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1 and df[col].nunique() < 100:
                categorical_cols.append(col)

    return categorical_cols


def preprocess_train_data(
    train_df: pd.DataFrame,
    target_col: str = "target",
    missing_threshold: float = MISSING_THRESHOLD,
    clip_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """학습 데이터 전처리"""
    steps = ["target 분리", "고결측 컬럼 제거", "컬럼명 정리", "범주형 식별", "결측 지표", "좌표 보간", "결측 대체", "이상치 클리핑", "로그 변환"]
    bar = tqdm(steps, desc="학습 전처리", unit="단계", leave=True, colour="blue")

    # target 분리
    bar.set_description("학습 전처리 - target 분리")
    if target_col not in train_df.columns:
        raise ValueError(f"target 컬럼 '{target_col}'을 찾을 수 없습니다.")
    y = train_df[target_col].copy()
    X = train_df.drop(columns=[target_col])
    tqdm.write(f"  원본 데이터 크기: {X.shape}")
    bar.update(1)

    # 좌표 컬럼 보존
    preserve_cols = [col for col in ["좌표X", "좌표Y", "좌표x", "좌표y"] if col in X.columns]

    # 고결측 컬럼 제거
    bar.set_description("학습 전처리 - 고결측 컬럼 제거")
    X = remove_high_missing_columns(X, threshold=missing_threshold, preserve_cols=preserve_cols)
    bar.update(1)

    # 컬럼명 정리 (LightGBM 호환)
    bar.set_description("학습 전처리 - 컬럼명 정리")
    X = sanitize_column_names(X)
    bar.update(1)

    # 범주형 식별
    bar.set_description("학습 전처리 - 범주형 식별")
    categorical_cols = identify_categorical_columns(X)
    tqdm.write(f"  범주형 컬럼 수: {len(categorical_cols)}")
    bar.update(1)

    # 결측 지표 피처 (결측 대체 전에 계산)
    bar.set_description("학습 전처리 - 결측 지표")
    X["missing_count"] = X.isnull().sum(axis=1)
    bar.update(1)

    # 좌표 결측 대체 (시군구 평균 좌표로 보간)
    bar.set_description("학습 전처리 - 좌표 보간")
    coord_cols = [c for c in ["좌표X", "좌표Y"] if c in X.columns]
    if coord_cols and "시군구" in X.columns:
        for col in coord_cols:
            group_mean = X.groupby("시군구")[col].transform("mean")
            X[col] = X[col].fillna(group_mean)
            X[col] = X[col].fillna(X[col].mean())  # 그룹 평균도 없는 경우
        tqdm.write(f"  좌표 결측 보간 완료: {coord_cols}")
    bar.update(1)

    # 결측 대체
    bar.set_description("학습 전처리 - 결측값 대체")
    X = handle_missing_values(X, categorical_cols)
    bar.update(1)

    # 이상치 클리핑 (주요 수치 피처)
    bar.set_description("학습 전처리 - 이상치 클리핑")
    numeric_outlier_cols = [c for c in ["전용면적", "층", "건축년도", "주차대수", "k-전체세대수"] if c in X.columns]
    if numeric_outlier_cols:
        X = clip_outliers(X, numeric_outlier_cols, method="iqr", factor=3.0)
        tqdm.write(f"  수치형 이상치 클리핑: {len(numeric_outlier_cols)}개 컬럼 (IQR x3.0)")
    if clip_target:
        y = clip_outliers(pd.DataFrame({target_col: y}), [target_col], method="iqr", factor=2.0)[target_col]
        tqdm.write("  Target 이상치 클리핑: IQR x2.0")
    bar.update(1)

    # 로그 변환
    bar.set_description("학습 전처리 - 로그 변환")
    y_log = log_transform_target(y)
    bar.update(1)

    bar.set_description("학습 전처리 완료")
    bar.close()
    tqdm.write(f"  전처리 완료. 최종 데이터 크기: {X.shape}")

    return X, y_log, categorical_cols


def preprocess_test_data(
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    missing_threshold: float = MISSING_THRESHOLD,
) -> pd.DataFrame:
    """테스트 데이터 전처리"""
    steps = ["고결측 컬럼 제거", "컬럼명 정리", "결측 지표", "좌표 보간", "결측값 대체"]
    bar = tqdm(steps, desc="테스트 전처리", unit="단계", leave=True, colour="blue")

    X_test = test_df.copy()
    tqdm.write(f"  테스트 원본 데이터 크기: {X_test.shape}")

    preserve_cols = [col for col in ["좌표X", "좌표Y", "좌표x", "좌표y"] if col in X_test.columns]

    bar.set_description("테스트 전처리 - 고결측 컬럼 제거")
    X_test = remove_high_missing_columns(X_test, threshold=missing_threshold, preserve_cols=preserve_cols)
    bar.update(1)

    # 컬럼명 정리 (LightGBM 호환)
    bar.set_description("테스트 전처리 - 컬럼명 정리")
    X_test = sanitize_column_names(X_test)
    bar.update(1)

    # 결측 지표 피처
    bar.set_description("테스트 전처리 - 결측 지표")
    X_test["missing_count"] = X_test.isnull().sum(axis=1)
    bar.update(1)

    # 좌표 결측 대체 (시군구 평균 좌표)
    bar.set_description("테스트 전처리 - 좌표 보간")
    coord_cols = [c for c in ["좌표X", "좌표Y"] if c in X_test.columns]
    if coord_cols and "시군구" in X_test.columns:
        for col in coord_cols:
            group_mean = X_test.groupby("시군구")[col].transform("mean")
            X_test[col] = X_test[col].fillna(group_mean)
            X_test[col] = X_test[col].fillna(X_test[col].mean())
    bar.update(1)

    bar.set_description("테스트 전처리 - 결측값 대체")
    X_test = handle_missing_values(X_test, categorical_cols)
    bar.update(1)

    bar.set_description("테스트 전처리 완료")
    bar.close()
    tqdm.write(f"  전처리 완료. 최종 데이터 크기: {X_test.shape}")

    return X_test


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """학습/검증 데이터 분할"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
