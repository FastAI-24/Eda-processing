"""
제출용 파일(submission.csv) 생성 유틸리티

대회 제출 포맷: Id, target
- Id: 테스트 행 식별자 (1-based)
- target: 예측 가격 (만원, 원본 스케일)

모델이 log1p 스케일로 예측한 경우 inverse_transform(expm1)으로 역변환 필요.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_submission(
    predictions: np.ndarray | pd.Series,
    test_df: pd.DataFrame | None = None,
    n_test: int | None = None,
    id_offset: int = 1,
) -> pd.DataFrame:
    """제출용 DataFrame을 생성합니다.

    Args:
        predictions: 예측값 (원본 스케일, 만원). log1p 역변환 완료된 상태여야 함.
        test_df: 테스트 데이터 (행 수 확인용). None이면 n_test 필요.
        n_test: 테스트 행 수 (test_df 미제공 시 사용)
        id_offset: Id 시작 값 (기본 1, Kaggle convention)

    Returns:
        Id, target 컬럼을 가진 제출용 DataFrame

    Example:
        >>> pred_original = np.expm1(model.predict(X_test))  # log → 원본 스케일
        >>> sub = create_submission(pred_original, test_df=X_test)
        >>> sub.to_csv("submission.csv", index=False)
    """
    n = len(predictions) if hasattr(predictions, "__len__") else n_test
    if n is None or n == 0:
        if test_df is not None:
            n = len(test_df)
        else:
            raise ValueError("predictions 길이 또는 test_df, n_test 중 하나가 필요합니다.")

    ids = np.arange(id_offset, n + id_offset, dtype=np.int64)
    return pd.DataFrame({"Id": ids, "target": np.asarray(predictions).ravel()})


def create_baseline_submission(
    y_train: pd.Series | np.ndarray,
    n_test: int,
    id_offset: int = 1,
) -> pd.DataFrame:
    """기본값(median)으로 제출용 파일을 생성합니다.

    모델 없이 형식 검증용으로 사용합니다.
    y_train은 원본 스케일(만원)이어야 합니다.

    Args:
        y_train: 학습 타겟 (원본 스케일)
        n_test: 테스트 행 수
        id_offset: Id 시작 값

    Returns:
        Id, target 컬럼을 가진 제출용 DataFrame
    """
    baseline = np.median(y_train)
    preds = np.full(n_test, baseline, dtype=np.float64)
    return create_submission(preds, n_test=n_test, id_offset=id_offset)


def inverse_log_transform(y_log: np.ndarray | pd.Series) -> np.ndarray:
    """log1p 변환을 역변환합니다 (예측 시 사용).

    모델이 log1p(y) 스케일로 예측한 경우 원본 가격(만원)으로 복원합니다.
    """
    return np.expm1(np.asarray(y_log))
