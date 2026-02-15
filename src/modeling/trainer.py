"""
모델 학습 파이프라인 (Trainer)

K-Fold 교차 검증과 RMSE 평가를 수행하는 핵심 모듈입니다.

디자인 패턴:
    - Strategy Pattern: BaseModel 인터페이스를 통해 모델 교체 가능
    - Template Method: train_with_cv()가 학습/평가의 전체 흐름을 제어
    - Mediator Pattern: TrainingResult가 결과를 중앙 집약

개선 사항 (v2):
    - TimeSeriesSplit CV 지원 (계약년월 정렬 기반)
    - Fold 내 Target Encoding (CV 누수 방지)
    - 시간 기반 Sample Weight (최근 데이터 가중치)

사용 예:
    from modeling import Trainer, LightGBMModel, ModelConfig

    config = ModelConfig(n_splits=5, cv_strategy="timeseries")
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
from sklearn.model_selection import KFold, TimeSeriesSplit
from tqdm import tqdm

from .base import BaseModel, TrainingResult
from .config import ModelConfig


# ─────────────────────────────────────────────────────────────
# Fold 내 Target Encoding 유틸리티
# ─────────────────────────────────────────────────────────────
def compute_fold_target_encoding(
    X_train_fold: pd.DataFrame,
    y_train_fold: np.ndarray,
    X_val_fold: pd.DataFrame,
    X_test: pd.DataFrame | None,
    te_cols: list[str],
    smoothing: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Fold 내부에서 Bayesian Smoothed Target Encoding을 수행합니다.

    학습 fold에서만 encoding map을 계산하여 CV 누수를 방지합니다.
    """
    X_train_fold = X_train_fold.copy()
    X_val_fold = X_val_fold.copy()
    X_test_out = X_test.copy() if X_test is not None else None

    global_mean = float(np.mean(y_train_fold))

    for col in te_cols:
        if col not in X_train_fold.columns:
            continue

        # 그룹별 통계 (train fold만 사용)
        tmp = pd.DataFrame({
            "key": X_train_fold[col].astype(str),
            "y": y_train_fold,
        })
        group_stats = tmp.groupby("key")["y"].agg(["mean", "count"])

        # Bayesian Smoothed Mean
        smoothed = (
            group_stats["count"] * group_stats["mean"]
            + smoothing * global_mean
        ) / (group_stats["count"] + smoothing)
        encoding_map = smoothed.to_dict()

        # 적용
        te_col = f"te_{col}"
        X_train_fold[te_col] = (
            X_train_fold[col].astype(str).map(encoding_map).fillna(global_mean)
        )
        X_val_fold[te_col] = (
            X_val_fold[col].astype(str).map(encoding_map).fillna(global_mean)
        )
        if X_test_out is not None:
            X_test_out[te_col] = (
                X_test_out[col].astype(str).map(encoding_map).fillna(global_mean)
            )

        # 빈도 인코딩
        freq_col = f"freq_{col}"
        freq_map = X_train_fold[col].astype(str).value_counts().to_dict()
        X_train_fold[freq_col] = (
            X_train_fold[col].astype(str).map(freq_map).fillna(0).astype("float64")
        )
        X_val_fold[freq_col] = (
            X_val_fold[col].astype(str).map(freq_map).fillna(0).astype("float64")
        )
        if X_test_out is not None:
            X_test_out[freq_col] = (
                X_test_out[col].astype(str).map(freq_map).fillna(0).astype("float64")
            )

    return X_train_fold, X_val_fold, X_test_out


# ─────────────────────────────────────────────────────────────
# Sample Weight 유틸리티
# ─────────────────────────────────────────────────────────────
def compute_sample_weight(
    X_fold: pd.DataFrame,
    decay: float = 0.05,
) -> np.ndarray | None:
    """시간 기반 지수 감쇠 샘플 가중치를 계산합니다.

    최근 데이터에 더 큰 가중치를 부여하여 시장 트렌드 변화를 반영합니다.

    Args:
        X_fold: 학습 fold 데이터 (계약년 컬럼 필요)
        decay: 지수 감쇠 계수 (클수록 최근 가중치↑)

    Returns:
        sample_weight 배열 또는 None (계약년 컬럼 없는 경우)
    """
    year_col = None
    for c in ["계약년", "계약년월"]:
        if c in X_fold.columns:
            year_col = c
            break

    if year_col is None:
        return None

    if year_col == "계약년월":
        years = pd.to_numeric(X_fold[year_col], errors="coerce") // 100
    else:
        years = pd.to_numeric(X_fold[year_col], errors="coerce")

    years = years.fillna(years.median())
    max_year = years.max()
    weights = np.exp(-decay * (max_year - years.values).astype(float))

    # 정규화 (평균 1.0)
    weights = weights / weights.mean()
    return weights


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────
class Trainer:
    """K-Fold 교차 검증 기반 모델 학습 및 RMSE 평가기.

    Template Method 패턴으로 학습 흐름을 제어합니다:
        1. 데이터 준비 (타겟 변환)
        2. K-Fold / TimeSeriesSplit 분할
        3. 각 Fold에서 (선택적) Target Encoding
        4. 시간 기반 Sample Weight 적용
        5. 모델 학습/검증
        6. RMSE 계산 및 집계
        7. OOF 예측 및 테스트 예측 생성
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()

    def _detect_categorical_features(self, X: pd.DataFrame) -> list[str]:
        """범주형 피처를 자동 감지합니다."""
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

    def _create_cv_splitter(self, X: pd.DataFrame):
        """CV 전략에 따라 splitter를 생성합니다."""
        cfg = self._config

        if cfg.cv_strategy == "timeseries":
            # 계약년월로 정렬된 인덱스 기반 TimeSeriesSplit
            return TimeSeriesSplit(n_splits=cfg.n_splits)
        else:
            return KFold(
                n_splits=cfg.n_splits,
                shuffle=True,
                random_state=cfg.random_state,
            )

    def train_with_cv(
        self,
        model_factory: type[BaseModel] | BaseModel,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        X_test: pd.DataFrame | None = None,
    ) -> TrainingResult:
        """K-Fold 교차 검증으로 모델을 학습하고 RMSE를 평가합니다."""
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

        # ── TimeSeriesSplit인 경우 정렬 ──
        sort_col = None
        if cfg.cv_strategy == "timeseries":
            for c in ["계약년월", "계약일자"]:
                if c in X.columns:
                    sort_col = c
                    break
            if sort_col:
                sort_idx = X[sort_col].argsort()
                X = X.iloc[sort_idx].reset_index(drop=True)
                if isinstance(y_transformed, np.ndarray):
                    y_transformed = y_transformed[sort_idx]
                else:
                    y_transformed = y_transformed.iloc[sort_idx].reset_index(drop=True)
                if isinstance(y, np.ndarray):
                    y = y[sort_idx]
                else:
                    y = y.iloc[sort_idx].reset_index(drop=True)
                print(f"TimeSeriesSplit: '{sort_col}' 기준 정렬 완료")

        # ── CV Splitter ──
        cv = self._create_cv_splitter(X)

        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test)) if X_test is not None else None
        fold_scores: list[float] = []
        all_importances: list[pd.DataFrame] = []
        trained_models: list = []

        # ── 요약 정보 출력 ──
        info_parts = [
            f"CV={cfg.cv_strategy}",
            f"K={cfg.n_splits}",
            f"데이터={X.shape}",
        ]
        if cfg.use_sample_weight:
            info_parts.append(f"SW(decay={cfg.sample_weight_decay})")
        if cfg.use_fold_target_encoding:
            info_parts.append(f"TE({len(cfg.target_encode_cols)}cols)")
        print(f"\n  ⚙ {model_name} | {' | '.join(info_parts)}")

        # ── tqdm Fold 진행 바 ──
        fold_bar = tqdm(
            enumerate(cv.split(X), 1),
            total=cfg.n_splits,
            desc=f"  🔄 {model_name}",
            bar_format="  {l_bar}{bar:30}{r_bar}",
            ncols=100,
        )

        for fold_idx, (train_idx, val_idx) in fold_bar:
            fold_start = time.time()

            X_train_fold = X.iloc[train_idx].copy()
            y_train_fold = (
                y_transformed[train_idx]
                if isinstance(y_transformed, np.ndarray)
                else y_transformed.iloc[train_idx]
            )
            X_val_fold = X.iloc[val_idx].copy()
            y_val_fold = (
                y_transformed[val_idx]
                if isinstance(y_transformed, np.ndarray)
                else y_transformed.iloc[val_idx]
            )

            fold_bar.set_postfix_str(
                f"Fold {fold_idx} | 학습:{len(train_idx):,} 검증:{len(val_idx):,}"
            )

            # ── Fold 내 Target Encoding ──
            X_test_fold = None
            if cfg.use_fold_target_encoding and cfg.target_encode_cols:
                y_train_np = (
                    y_train_fold
                    if isinstance(y_train_fold, np.ndarray)
                    else y_train_fold.values
                )
                X_train_fold, X_val_fold, X_test_fold = compute_fold_target_encoding(
                    X_train_fold,
                    y_train_np,
                    X_val_fold,
                    X_test,
                    cfg.target_encode_cols,
                    cfg.target_encode_smoothing,
                )
                if fold_idx == 1:
                    te_cols_added = [
                        c for c in X_train_fold.columns
                        if c.startswith("te_") or c.startswith("freq_")
                    ]
                    tqdm.write(f"    TE 피처: {len(te_cols_added)}개 생성")

            # ── Sample Weight ──
            weights = None
            if cfg.use_sample_weight:
                weights = compute_sample_weight(X_train_fold, cfg.sample_weight_decay)

            # ── 모델 인스턴스 생성 ──
            if isinstance(model_factory, type):
                model = model_factory(cfg)
            else:
                model = type(model_factory)(cfg)

            # ── 학습 ──
            model.train(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                categorical_features=cat_features,
                sample_weight=weights,
            )

            # ── 검증 예측 및 RMSE ──
            val_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            fold_scores.append(rmse)

            # ── OOF 예측 저장 ──
            oof_preds[val_idx] = val_pred

            # ── 테스트 예측 누적 ──
            if X_test is not None and test_preds is not None:
                X_test_for_pred = X_test_fold if X_test_fold is not None else X_test
                test_preds += model.predict(X_test_for_pred) / cfg.n_splits

            # ── 피처 중요도 ──
            imp = model.get_feature_importance()
            if imp is not None:
                imp["fold"] = fold_idx
                all_importances.append(imp)

            trained_models.append(model)

            fold_elapsed = time.time() - fold_start
            avg_rmse = np.mean(fold_scores)
            fold_bar.set_postfix_str(
                f"Fold {fold_idx} RMSE={rmse:.6f} | 평균={avg_rmse:.6f} | {fold_elapsed:.0f}s"
            )

        # ── 최종 결과 집계 ──
        mean_rmse = np.mean(fold_scores)
        std_rmse = np.std(fold_scores)
        total_elapsed = time.time() - start_time

        # 전체 OOF RMSE
        oof_rmse = np.sqrt(mean_squared_error(y_transformed, oof_preds))

        tqdm.write(f"\n  ✅ {model_name} 완료")
        tqdm.write(f"     Fold별 : {[f'{s:.6f}' for s in fold_scores]}")
        tqdm.write(f"     평균   : {mean_rmse:.6f} (±{std_rmse:.6f})")
        tqdm.write(f"     OOF    : {oof_rmse:.6f}")
        tqdm.write(f"     시간   : {total_elapsed:.1f}초 ({total_elapsed / 60:.1f}분)")

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
