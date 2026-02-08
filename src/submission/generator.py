"""제출 파일 생성"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.config import SUBMISSION_DIR
from src.data.preprocessor import inverse_log_transform


def generate_submission(
    predictions: np.ndarray,
    sample_submission: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """제출 파일 생성
    
    Args:
        predictions: 예측값 (로그 변환된 값)
        sample_submission: 샘플 제출 파일 DataFrame
        output_path: 저장 경로 (None이면 기본 경로 사용)
    
    Returns:
        제출 파일 DataFrame
    """
    # 예측값을 원래 스케일로 복원 (expm1) → 정수 변환
    predictions_original = inverse_log_transform(predictions)
    predictions_int = np.round(predictions_original).astype(int)
    
    # 제출 파일 생성
    submission = sample_submission.copy()
    submission["target"] = predictions_int
    
    # 저장
    if output_path is None:
        output_path = SUBMISSION_DIR / "submission.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"\n제출 파일 생성 완료: {output_path}")
    print(f"예측값 통계 (정수 변환):")
    print(f"  평균: {predictions_int.mean():.0f}")
    print(f"  중앙값: {np.median(predictions_int):.0f}")
    print(f"  최소값: {predictions_int.min()}")
    print(f"  최대값: {predictions_int.max()}")
    
    return submission
