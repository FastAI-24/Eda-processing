"""
모델 학습 파이프라인 (Trainer)

K-Fold 교차 검증과 RMSE 평가를 수행하는 핵심 모듈입니다.

디자인 패턴:
    - Strategy Pattern: BaseModel 인터페이스를 통해 모델 교체 가능
    - Template Method: train_with_cv()가 학습/평가의 전체 흐름을 제어
    - Mediator Pattern: TrainingResult가 결과를 중앙 집약

사용 예:
    from modeling import Trainer, LightGBMModel, ModelConfig

    config = ModelConfig(n_splits=5)
    trainer = Trainer(config)

    model = LightGBMModel(config)
    result = trainer.train_with_cv(model, X_train, y_train)
    print(result.summary())
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from .base import BaseModel, TrainingResult
from .config import ModelConfig


class Trainer:
    """K-Fold 교차 검증 기반 모델 학습 및 RMSE 평가기.

    Template Method 패턴으로 학습 흐름을 제어합니다:
        1. 데이터 준비 (타겟 변환)
        2. K-Fold 분할
        3. 각 Fold에서 학습/검증
        4. RMSE 계산 및 집계
        5. OOF 예측 및 테스트 예측 생성
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()

    def _detect_categorical_features(self, X: pd.DataFrame) -> list[str]:
        """범주형 피처를 자동 감지합니다.

        pandas 3.0+에서는 문자열이 str dtype으로 로드되므로
        object, string, category 모두 범주형으로 감지합니다.
        """
        if self._config.categorical_features is not None:
            return [c for c in self._config.categorical_features if c in X.columns]

        cat_cols = []
        for col in X.columns:
            dtype_str = str(X[col].dtype)
            if (
                X[col].dtype == object
                or dtype_str in ("category", "string", "str", "object")
                or pd.api.types.is_string_dtype(X[col])
            ):
                cat_cols.append(col)
        return cat_cols

    def train_with_cv(
        self,
        model_factory: type[BaseModel] | BaseModel,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        X_test: pd.DataFrame | None = None,
    ) -> TrainingResult:
        """K-Fold 교차 검증으로 모델을 학습하고 RMSE를 평가합니다.

        Args:
            model_factory: BaseModel 클래스 또는 인스턴스.
                클래스가 전달되면 각 Fold마다 새 인스턴스를 생성합니다.
                인스턴스가 전달되면 이름과 설정을 참조하되, 각 Fold는 새 인스턴스로 학습합니다.
            X: 학습 피처 (전체)
            y: 학습 타겟 (원본 스케일, 만원)
            X_test: 테스트 피처 (None이면 테스트 예측 건너뜀)

        Returns:
            TrainingResult: 교차 검증 결과 (RMSE, OOF 예측, 테스트 예측 등)
        """
        cfg = self._config
        start_time = time.time()

        # ── 타겟 변환 ──
        if cfg.use_log_target:
            y_transformed = np.log1p(y)
            print(f"타겟 변환: log1p (범위: [{y_transformed.min():.4f}, {y_transformed.max():.4f}])")
        else:
            y_transformed = np.asarray(y, dtype=np.float64)
            print(f"타겟: 원본 스케일 (범위: [{y_transformed.min():,.0f}, {y_transformed.max():,.0f}])")

        # ── 모델 정보 ──
        if isinstance(model_factory, type):
            sample_model = model_factory(cfg)
            model_name = sample_model.name
        else:
            model_name = model_factory.name

        # ── 범주형 피처 감지 ──
        cat_features = self._detect_categorical_features(X)
        print(f"범주형 피처: {len(cat_features)}개")

        # ── K-Fold 설정 ──
        kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test)) if X_test is not None else None
        fold_scores: list[float] = []
        all_importances: list[pd.DataFrame] = []
        trained_models: list = []

        print(f"\n{'='*60}")
        print(f"모델 학습 시작: {model_name}")
        print(f"K-Fold: {cfg.n_splits}  |  데이터: {X.shape}  |  피처: {X.shape[1]}")
        print(f"{'='*60}")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            fold_start = time.time()
            print(f"\n{'─'*60}")
            print(f"[Fold {fold_idx}/{cfg.n_splits}]")
            print(f"{'─'*60}")
            print(f"  학습: {len(train_idx):,}건  |  검증: {len(val_idx):,}건")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y_transformed[train_idx] if isinstance(y_transformed, np.ndarray) else y_transformed.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y_transformed[val_idx] if isinstance(y_transformed, np.ndarray) else y_transformed.iloc[val_idx]

            # 모델 인스턴스 생성 (각 Fold마다 새로 생성)
            if isinstance(model_factory, type):
                model = model_factory(cfg)
            else:
                model = type(model_factory)(cfg)

            # 학습
            model.train(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                categorical_features=cat_features,
            )

            # 검증 예측 및 RMSE 계산
            val_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            fold_scores.append(rmse)
            print(f"  Fold {fold_idx} RMSE: {rmse:.6f}")

            # OOF 예측 저장
            oof_preds[val_idx] = val_pred

            # 테스트 예측 누적 (평균 앙상블)
            if X_test is not None and test_preds is not None:
                test_preds += model.predict(X_test) / cfg.n_splits

            # 피처 중요도
            imp = model.get_feature_importance()
            if imp is not None:
                imp["fold"] = fold_idx
                all_importances.append(imp)

            trained_models.append(model)

            fold_elapsed = time.time() - fold_start
            print(f"  소요 시간: {fold_elapsed:.1f}초")

        # ── 최종 결과 집계 ──
        mean_rmse = np.mean(fold_scores)
        std_rmse = np.std(fold_scores)
        total_elapsed = time.time() - start_time

        # 전체 OOF RMSE
        oof_rmse = np.sqrt(mean_squared_error(y_transformed, oof_preds))

        print(f"\n{'='*60}")
        print(f"학습 완료: {model_name}")
        print(f"{'='*60}")
        print(f"  Fold별 RMSE: {[f'{s:.6f}' for s in fold_scores]}")
        print(f"  평균 RMSE:   {mean_rmse:.6f} (±{std_rmse:.6f})")
        print(f"  OOF RMSE:    {oof_rmse:.6f}")
        print(f"  총 소요 시간: {total_elapsed:.1f}초")

        # 피처 중요도 집계
        feature_importances = None
        if all_importances:
            combined = pd.concat(all_importances, ignore_index=True)
            feature_importances = (
                combined.groupby("feature")["importance"]
                .mean()
                .reset_index()
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        return TrainingResult(
            model_name=model_name,
            fold_scores=fold_scores,
            mean_rmse=mean_rmse,
            std_rmse=std_rmse,
            oof_predictions=oof_preds,
            test_predictions=test_preds,
            feature_importances=feature_importances,
            trained_models=trained_models,
        )

