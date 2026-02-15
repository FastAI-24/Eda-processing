"""
가중 평균 앙상블 (Weighted Average Ensemble)

LightGBM, XGBoost, CatBoost의 예측을 OOF RMSE 역수 기반으로
가중 평균하여 최종 예측을 생성합니다.

사용 예:
    from modeling.ensemble import EnsembleTrainer
    from modeling.config import ModelConfig

    config = ModelConfig()
    trainer = EnsembleTrainer(config)
    result = trainer.train_ensemble(X_train, y_train, X_test)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import TrainingResult
from .config import ModelConfig
from .models import CatBoostModel, LightGBMModel, XGBoostModel
from .trainer import Trainer


class EnsembleTrainer:
    """다중 모델 가중 평균 앙상블 학습기."""

    MODEL_MAP = {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "catboost": CatBoostModel,
    }

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()

    def train_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        X_test: pd.DataFrame | None = None,
    ) -> dict:
        """각 모델을 CV로 학습하고 가중 평균 앙상블을 수행합니다.

        Returns:
            {
                "results": {model_name: TrainingResult},
                "weights": {model_name: float},
                "ensemble_test_predictions": np.ndarray | None,
                "ensemble_oof_predictions": np.ndarray,
                "ensemble_oof_rmse": float,
            }
        """
        cfg = self._config
        trainer = Trainer(cfg)
        total_start = time.time()

        results: dict[str, TrainingResult] = {}
        n_models = len(cfg.ensemble_models)

        print(f"\n{'━'*60}")
        print(f"  🏗  앙상블 파이프라인 ({n_models}개 모델 × {cfg.n_splits}-Fold CV)")
        print(f"{'━'*60}")

        # ── 모델별 학습 진행 바 ──
        model_bar = tqdm(
            cfg.ensemble_models,
            desc="  📊 앙상블 진행",
            bar_format="  {l_bar}{bar:30}{r_bar}",
            ncols=100,
        )

        for model_key in model_bar:
            model_cls = self.MODEL_MAP.get(model_key)
            if model_cls is None:
                tqdm.write(f"  ⚠ 알 수 없는 모델: {model_key} — 건너뜀")
                continue

            model_bar.set_postfix_str(f"{model_key.upper()} 학습 중...")

            try:
                result = trainer.train_with_cv(model_cls, X, y, X_test)
                results[model_key] = result
                model_bar.set_postfix_str(
                    f"{model_key.upper()} RMSE={result.mean_rmse:.6f}"
                )
            except Exception as e:
                tqdm.write(f"  ❌ {model_key} 학습 실패: {e}")
                continue

        if not results:
            raise RuntimeError("모든 모델 학습이 실패했습니다.")

        # ── 가중치 계산 (OOF RMSE 역수) ──
        inv_rmse = {}
        for name, res in results.items():
            inv_rmse[name] = 1.0 / (res.mean_rmse + 1e-10)

        total_inv = sum(inv_rmse.values())
        weights = {name: v / total_inv for name, v in inv_rmse.items()}

        # ── 가중 평균 예측 ──
        if cfg.use_log_target:
            y_transformed = np.log1p(y)
        else:
            y_transformed = np.asarray(y, dtype=np.float64)

        oof_pred = np.zeros(len(X))
        for name, res in results.items():
            if res.oof_predictions is not None:
                oof_pred += weights[name] * res.oof_predictions

        from sklearn.metrics import mean_squared_error
        oof_rmse = np.sqrt(mean_squared_error(y_transformed, oof_pred))

        test_pred = None
        if X_test is not None:
            test_pred = np.zeros(len(X_test))
            for name, res in results.items():
                if res.test_predictions is not None:
                    test_pred += weights[name] * res.test_predictions

        total_elapsed = time.time() - total_start

        # ── 최종 결과 요약 ──
        print(f"\n{'━'*60}")
        print(f"  📋 앙상블 결과 요약")
        print(f"{'━'*60}")
        for name, res in results.items():
            w = weights[name]
            print(f"  {'├' if name != list(results.keys())[-1] else '└'}"
                  f" {name:<12s}  RMSE={res.mean_rmse:.6f} (±{res.std_rmse:.6f})  가중치={w:.4f}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  🏆 앙상블 OOF RMSE = {oof_rmse:.6f}")
        print(f"  ⏱  총 소요 시간: {total_elapsed:.1f}초 ({total_elapsed / 60:.1f}분)")
        print(f"{'━'*60}")

        return {
            "results": results,
            "weights": weights,
            "ensemble_test_predictions": test_pred,
            "ensemble_oof_predictions": oof_pred,
            "ensemble_oof_rmse": oof_rmse,
        }
