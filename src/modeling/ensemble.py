"""
앙상블 학습기 (Weighted Average / Stacking / Multi-seed)

LightGBM, XGBoost, CatBoost의 예측을 다음 전략으로 결합합니다:
- weighted: OOF RMSE 역수 기반 가중 평균 (기본)
- stacking: Ridge 메타 학습기로 2단계 앙상블 (Exp10)
- multi-seed: 같은 모델 다른 시드로 예측 평균화 (Exp10)

사용 예:
    from modeling.ensemble import EnsembleTrainer
    from modeling.config import ModelConfig

    config = ModelConfig()
    config.ensemble_strategy = "stacking"  # 또는 "weighted"
    config.ensemble_use_multi_seed = True
    trainer = EnsembleTrainer(config)
    result = trainer.train_ensemble(X_train, y_train, X_test)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm import tqdm

from .base import TrainingResult
from .config import ModelConfig
from .models import CatBoostModel, LightGBMModel, LightGBMQuantileModel, MLPModel, XGBoostModel
from .trainer import Trainer


class EnsembleTrainer:
    """다중 모델 가중 평균 앙상블 학습기."""

    MODEL_MAP = {
        "lightgbm": LightGBMModel,
        "lightgbm_quantile": LightGBMQuantileModel,
        "xgboost": XGBoostModel,
        "catboost": CatBoostModel,
        "mlp": MLPModel,
    }

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()

    def train_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        X_test: pd.DataFrame | None = None,
    ) -> dict:
        """각 모델을 CV로 학습하고 앙상블을 수행합니다.

        ensemble_strategy:
            - "weighted": OOF RMSE 역수 가중 평균
            - "stacking": Ridge 메타 학습기 (2단계)
        ensemble_use_multi_seed: True면 각 모델을 여러 시드로 학습 후 평균

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
        total_start = time.time()

        results: dict[str, TrainingResult] = {}
        n_models = len(cfg.ensemble_models)
        seeds = cfg.ensemble_seeds if cfg.ensemble_use_multi_seed else [cfg.random_state]
        n_seeds = len(seeds)

        strategy_info = f"{cfg.ensemble_strategy}"
        if cfg.ensemble_use_multi_seed:
            strategy_info += f" + Multi-seed({n_seeds})"

        print(f"\n{'━'*60}")
        print(f"  🏗  앙상블 파이프라인 ({n_models}개 모델 × {cfg.n_splits}-Fold × {strategy_info})")
        print(f"{'━'*60}")

        # ── 모델별 (× 시드별) 학습 ──
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

            oof_list: list[np.ndarray] = []
            test_list: list[np.ndarray] = []
            fold_scores_list: list[list[float]] = []
            last_result: TrainingResult | None = None

            for seed_idx, seed in enumerate(seeds):
                seed_cfg = cfg.with_seed(seed) if seed != cfg.random_state else cfg
                trainer = Trainer(seed_cfg)
                model_bar.set_postfix_str(
                    f"{model_key.upper()} seed={seed}" if n_seeds > 1 else f"{model_key.upper()} 학습 중..."
                )

                try:
                    result = trainer.train_with_cv(model_cls, X, y, X_test)
                    last_result = result
                    if result.oof_predictions is not None:
                        oof_list.append(result.oof_predictions)
                    if result.test_predictions is not None and X_test is not None:
                        test_list.append(result.test_predictions)
                    fold_scores_list.append(result.fold_scores)
                    if n_seeds == 1:
                        results[model_key] = result
                        model_bar.set_postfix_str(f"{model_key.upper()} RMSE={result.mean_rmse:.6f}")
                except Exception as e:
                    tqdm.write(f"  ❌ {model_key} (seed={seed}) 학습 실패: {e}")
                    continue

            if not oof_list:
                continue

            # Multi-seed: 시드별 예측 평균
            if n_seeds > 1 and last_result is not None:
                oof_avg = np.mean(oof_list, axis=0)
                test_avg = np.mean(test_list, axis=0) if test_list else None
                mean_rmse = float(np.mean([np.mean(s) for s in fold_scores_list]))
                std_rmse = float(np.std([np.mean(s) for s in fold_scores_list]))
                results[model_key] = TrainingResult(
                    model_name=model_key,
                    fold_scores=fold_scores_list[0],
                    mean_rmse=mean_rmse,
                    std_rmse=std_rmse,
                    oof_predictions=oof_avg,
                    test_predictions=test_avg,
                    feature_importances=last_result.feature_importances,
                    trained_models=[],
                )
                model_bar.set_postfix_str(f"{model_key.upper()} RMSE={mean_rmse:.6f} (×{n_seeds} seeds)")

        if not results:
            raise RuntimeError("모든 모델 학습이 실패했습니다.")

        # ── 타겟 변환 ──
        if cfg.use_log_target:
            y_transformed = np.log1p(y)
        else:
            y_transformed = np.asarray(y, dtype=np.float64)

        # ── 앙상블 전략에 따른 최종 예측 ──
        if cfg.ensemble_strategy == "stacking":
            oof_pred, test_pred, weights = self._stacking_ensemble(
                results, y_transformed, X_test is not None
            )
        else:
            inv_rmse = {n: 1.0 / (r.mean_rmse + 1e-10) for n, r in results.items()}
            total_inv = sum(inv_rmse.values())
            weights = {n: v / total_inv for n, v in inv_rmse.items()}

            oof_pred = np.zeros(len(X))
            for name, res in results.items():
                if res.oof_predictions is not None:
                    oof_pred += weights[name] * res.oof_predictions

            test_pred = None
            if X_test is not None:
                test_pred = np.zeros(len(X_test))
                for name, res in results.items():
                    if res.test_predictions is not None:
                        test_pred += weights[name] * res.test_predictions

        from sklearn.metrics import mean_squared_error
        oof_rmse = np.sqrt(mean_squared_error(y_transformed, oof_pred))

        total_elapsed = time.time() - total_start

        # ── 최종 결과 요약 ──
        print(f"\n{'━'*60}")
        print(f"  📋 앙상블 결과 요약 ({cfg.ensemble_strategy})")
        print(f"{'━'*60}")
        for name, res in results.items():
            w = weights.get(name, 0.0)
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

    def _stacking_ensemble(
        self,
        results: dict[str, TrainingResult],
        y: np.ndarray,
        has_test: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
        """Ridge 메타 학습기로 2단계 Stacking 앙상블."""
        oof_matrix = np.column_stack([
            r.oof_predictions for r in results.values()
            if r.oof_predictions is not None
        ])
        meta = Ridge(alpha=1.0, random_state=42)
        meta.fit(oof_matrix, y)

        oof_pred = meta.predict(oof_matrix)
        test_pred = None
        if has_test:
            test_matrix = np.column_stack([
                r.test_predictions for r in results.values()
                if r.test_predictions is not None
            ])
            test_pred = meta.predict(test_matrix)

        # 가중치 = 메타 모델 계수 (해석용)
        coef = np.abs(meta.coef_)
        weights = {n: c for n, c in zip(results.keys(), coef / (coef.sum() + 1e-10))}
        return oof_pred, test_pred, weights
