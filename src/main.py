"""메인 실행 스크립트"""
import argparse
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

from src.config import (
    MODELS_DIR,
    RANDOM_SEED,
    OPTUNA_N_TRIALS,
    IS_REMOTE,
    LGBM_DEFAULT_PARAMS,
    N_FOLDS,
    MULTI_SEED_LIST,
    SAMPLE_WEIGHT_MIN,
    SAMPLE_WEIGHT_MAX,
)
from src.data.loader import load_all_data, load_sample_submission, load_test_data
from src.data.preprocessor import (
    preprocess_train_data,
    preprocess_test_data,
)
from src.features.engineer import (
    engineer_features,
    KFoldTargetEncoder,
    compute_region_stats,
    apply_region_stats,
    save_region_stats,
    load_region_stats,
    compute_time_lag_stats,
    apply_time_lag_stats,
    save_time_lag_stats,
    load_time_lag_stats,
)
from src.models.lightgbm_model import (
    LightGBMModel,
    cross_validate,
    cross_validate_and_predict,
    multi_seed_cross_validate_and_predict,
    optimize_hyperparameters,
    train_final_model,
)
from src.submission.generator import generate_submission


MODEL_FILENAME = "lightgbm_model.txt"
TARGET_ENCODER_FILENAME = "target_encoder.pkl"
REGION_STATS_FILENAME = "region_stats.pkl"
TIME_LAG_STATS_FILENAME = "time_lag_stats.pkl"


def main():
    parser = argparse.ArgumentParser(description="아파트 가격 예측 ML 파이프라인 (LightGBM)")
    parser.add_argument("--gpu", action="store_true", help="GPU 사용")
    parser.add_argument("--tune", action="store_true", help="하이퍼파라미터 튜닝 실행")
    parser.add_argument("--predict", action="store_true", help="예측 및 제출 파일 생성 (학습 후)")
    parser.add_argument("--predict-only", action="store_true", help="기존 모델로 예측만 실행")
    parser.add_argument("--model-path", type=str, default=None, help="모델 파일 경로")
    parser.add_argument("--n-trials", type=int, default=OPTUNA_N_TRIALS, help="Optuna 시도 횟수")
    parser.add_argument("--multi-seed", action="store_true", default=True, help="다중 시드 앙상블 (기본: 활성)")
    parser.add_argument("--no-multi-seed", dest="multi_seed", action="store_false", help="단일 시드 사용")

    args = parser.parse_args()

    use_gpu = args.gpu

    # 환경 정보 출력
    import platform
    env_label = "원격 서버 (고사양)" if IS_REMOTE else "로컬"
    print("=" * 50)
    print(f"환경: {env_label} | {platform.system()}/{platform.machine()}")
    print(f"LightGBM 파라미터: n_estimators={LGBM_DEFAULT_PARAMS['n_estimators']}, "
          f"num_leaves={LGBM_DEFAULT_PARAMS['num_leaves']}, "
          f"lr={LGBM_DEFAULT_PARAMS['learning_rate']}")

    if use_gpu:
        system = platform.system()
        machine = platform.machine()
        if system == "Darwin" and machine == "arm64":
            print("GPU: Apple Silicon (OpenCL)")
        elif system == "Linux":
            print("GPU: NVIDIA CUDA/OpenCL")
        else:
            print(f"GPU: {system}/{machine}")
        print("GPU 실패 시 자동으로 CPU 모드로 전환됩니다.")
    else:
        print("GPU: 비활성 (CPU 모드)")
    print("=" * 50)

    # ── 예측 전용 모드 ──
    if args.predict_only:
        _run_predict_only(args, use_gpu)
        return

    # ── 학습 파이프라인 ──
    _run_training_pipeline(args, use_gpu)


def _run_predict_only(args, use_gpu: bool):
    """기존 모델로 예측만 실행"""
    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / MODEL_FILENAME
    if not model_path.exists():
        print(f"\n오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 모델을 학습하세요: python -m src.main --gpu")
        sys.exit(1)

    steps = [
        "데이터 로드", "전처리", "피처 엔지니어링",
        "Target Encoding", "지역 통계", "시간 통계",
        "모델 로드", "예측", "후처리", "제출 파일 생성",
    ]
    pipeline_bar = tqdm(steps, desc="전체 파이프라인", unit="단계", leave=True, colour="green")

    # 데이터 로드
    pipeline_bar.set_description("데이터 로드")
    test = load_test_data()
    _, _, bus, subway = load_all_data()
    train, _, _, _ = load_all_data()
    pipeline_bar.update(1)

    # 전처리
    pipeline_bar.set_description("전처리")
    _, _, categorical_cols = preprocess_train_data(train, clip_target=False)
    X_test = preprocess_test_data(test, categorical_cols)
    pipeline_bar.update(1)

    # 피처 엔지니어링
    pipeline_bar.set_description("피처 엔지니어링")
    X_test = engineer_features(X_test, bus_df=bus, subway_df=subway)
    pipeline_bar.update(1)

    # Target Encoding (로드)
    pipeline_bar.set_description("Target Encoding")
    te_path = MODELS_DIR / TARGET_ENCODER_FILENAME
    if te_path.exists():
        target_encoder = KFoldTargetEncoder()
        target_encoder.load(te_path)
        X_test = target_encoder.transform(X_test)
        tqdm.write("  Target Encoder 로드 완료")
    else:
        tqdm.write("  경고: Target Encoder 파일 없음 — 건너뜀")
    pipeline_bar.update(1)

    # 지역 통계 (로드)
    pipeline_bar.set_description("지역 통계")
    rs_path = MODELS_DIR / REGION_STATS_FILENAME
    if rs_path.exists():
        region_stats = load_region_stats(rs_path)
        X_test = apply_region_stats(X_test, region_stats)
        tqdm.write("  Region Stats 로드 완료")
    else:
        tqdm.write("  경고: Region Stats 파일 없음 — 건너뜀")
    pipeline_bar.update(1)

    # 시간 통계 (로드)
    pipeline_bar.set_description("시간 통계")
    tl_path = MODELS_DIR / TIME_LAG_STATS_FILENAME
    if tl_path.exists():
        time_lag_stats = load_time_lag_stats(tl_path)
        X_test = apply_time_lag_stats(X_test, time_lag_stats)
        tqdm.write("  Time Lag Stats 로드 완료")
    else:
        tqdm.write("  경고: Time Lag Stats 파일 없음 — 건너뜀")
    pipeline_bar.update(1)

    # 모델 로드
    pipeline_bar.set_description("모델 로드")
    model = LightGBMModel(use_gpu=use_gpu)
    model.load(model_path)
    pipeline_bar.update(1)

    # 예측
    pipeline_bar.set_description("예측")
    y_pred_log = model.predict(X_test)
    pipeline_bar.update(1)

    # 후처리
    pipeline_bar.set_description("후처리")
    y_pred_log = _postprocess_predictions(y_pred_log)
    pipeline_bar.update(1)

    # 제출 파일 생성
    pipeline_bar.set_description("제출 파일 생성")
    sample_submission = load_sample_submission()
    generate_submission(y_pred_log, sample_submission)
    pipeline_bar.update(1)

    pipeline_bar.set_description("완료")
    pipeline_bar.close()
    print("\n" + "=" * 50)
    print("예측 완료!")
    print("=" * 50)


def _compute_sample_weight(X_train: np.ndarray) -> np.ndarray:
    """시간 기반 샘플 가중치 계산: 최신 데이터에 높은 가중치"""
    if "contract_year" not in X_train.columns:
        return None
    year = X_train["contract_year"].values.astype(float)
    year_min, year_max = year.min(), year.max()
    if year_max == year_min:
        return None
    # 선형 보간: 가장 오래된 = SAMPLE_WEIGHT_MIN, 최신 = SAMPLE_WEIGHT_MAX
    weights = SAMPLE_WEIGHT_MIN + (SAMPLE_WEIGHT_MAX - SAMPLE_WEIGHT_MIN) * (
        (year - year_min) / (year_max - year_min)
    )
    tqdm.write(
        f"  샘플 가중치: {int(year_min)}년={weights.min():.2f} ~ "
        f"{int(year_max)}년={weights.max():.2f}"
    )
    return weights


def _run_training_pipeline(args, use_gpu: bool):
    """전체 학습 파이프라인"""
    steps = [
        "데이터 로드", "전처리", "피처 엔지니어링",
        "Target Encoding", "지역 통계", "시간 통계",
        "샘플 가중치",
    ]
    if args.tune:
        steps.append("하이퍼파라미터 튜닝")
    if args.multi_seed:
        steps.append("다중 시드 앙상블")
    else:
        steps.append("K-Fold OOF 예측")
    steps += ["최종 모델 학습", "후처리", "제출 파일 생성"]

    pipeline_bar = tqdm(steps, desc="전체 파이프라인", unit="단계", leave=True, colour="green")

    # ── 1) 데이터 로드 ──
    pipeline_bar.set_description("데이터 로드")
    train, test, bus, subway = load_all_data()
    tqdm.write(f"학습 데이터: {train.shape}, 테스트 데이터: {test.shape}")
    pipeline_bar.update(1)

    # ── 2) 전처리 ──
    pipeline_bar.set_description("전처리")
    X_train, y_train_log, categorical_cols = preprocess_train_data(train)
    X_test = preprocess_test_data(test, categorical_cols)
    pipeline_bar.update(1)

    # ── 3) 피처 엔지니어링 ──
    pipeline_bar.set_description("피처 엔지니어링")
    X_train = engineer_features(X_train, bus_df=bus, subway_df=subway)
    X_test = engineer_features(X_test, bus_df=bus, subway_df=subway)

    # 새로운 범주형 컬럼 추가 (시군구 파싱 + k-* 피처)
    new_cat_cols = [
        c for c in [
            "시도", "구군", "동",
            "k-관리방식", "k-난방방식", "k-복도유형",
            "k-건설사_시공사", "k-세대타입_분양형태", "k-단지분류_아파트_주상복합등등",
        ]
        if c in X_train.columns
    ]
    categorical_cols = list(set(categorical_cols + new_cat_cols))

    pipeline_bar.update(1)

    # ── 4) Target Encoding ──
    pipeline_bar.set_description("Target Encoding")
    target_encoder = KFoldTargetEncoder()
    X_train = target_encoder.fit_transform(X_train, y_train_log)
    X_test = target_encoder.transform(X_test)
    target_encoder.save(MODELS_DIR / TARGET_ENCODER_FILENAME)
    pipeline_bar.update(1)

    # ── 5) 지역 통계 ──
    pipeline_bar.set_description("지역 통계")
    region_stats = compute_region_stats(X_train, y_train_log)
    X_train = apply_region_stats(X_train, region_stats)
    X_test = apply_region_stats(X_test, region_stats)
    save_region_stats(region_stats, MODELS_DIR / REGION_STATS_FILENAME)
    pipeline_bar.update(1)

    # ── 6) 시간 통계 ──
    pipeline_bar.set_description("시간 통계")
    time_lag_stats = compute_time_lag_stats(X_train, y_train_log)
    X_train = apply_time_lag_stats(X_train, time_lag_stats)
    X_test = apply_time_lag_stats(X_test, time_lag_stats)
    save_time_lag_stats(time_lag_stats, MODELS_DIR / TIME_LAG_STATS_FILENAME)
    pipeline_bar.update(1)

    # 공통 컬럼만 유지
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    categorical_cols = [col for col in categorical_cols if col in common_cols]
    tqdm.write(f"최종 피처 수: {len(common_cols)}, 범주형 피처 수: {len(categorical_cols)}")

    # ── 7) 샘플 가중치 ──
    pipeline_bar.set_description("샘플 가중치")
    sample_weight = _compute_sample_weight(X_train)
    pipeline_bar.update(1)

    # ── 8) 하이퍼파라미터 튜닝 (선택) ──
    if args.tune:
        pipeline_bar.set_description("하이퍼파라미터 튜닝")
        best_params = optimize_hyperparameters(
            X_train, y_train_log, categorical_cols,
            use_gpu=use_gpu, n_trials=args.n_trials,
        )
        pipeline_bar.update(1)
    else:
        best_params = {}

    # ── 9) K-Fold OOF 예측 (다중 시드 또는 단일) ──
    if args.multi_seed:
        pipeline_bar.set_description("다중 시드 앙상블")
        cv_results, y_pred_log = multi_seed_cross_validate_and_predict(
            X_train, y_train_log, X_test, categorical_cols,
            use_gpu=use_gpu, n_folds=N_FOLDS,
            seeds=MULTI_SEED_LIST,
            sample_weight=sample_weight,
            **best_params,
        )
    else:
        pipeline_bar.set_description("K-Fold OOF 예측")
        cv_results, y_pred_log = cross_validate_and_predict(
            X_train, y_train_log, X_test, categorical_cols,
            use_gpu=use_gpu, n_folds=N_FOLDS,
            sample_weight=sample_weight,
            **best_params,
        )
    pipeline_bar.update(1)

    # ── 10) 최종 모델 학습 (predict-only 모드용 저장) ──
    pipeline_bar.set_description("최종 모델 학습")
    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / MODEL_FILENAME
    final_model = train_final_model(
        X_train, y_train_log, categorical_cols,
        use_gpu=use_gpu, model_params=best_params, save_path=model_path,
        sample_weight=sample_weight,
    )

    # 피처 중요도 출력
    _print_feature_importance(final_model)
    pipeline_bar.update(1)

    # ── 11) 후처리 ──
    pipeline_bar.set_description("후처리")
    y_pred_log = _postprocess_predictions(y_pred_log, y_train_log)
    pipeline_bar.update(1)

    # ── 12) 제출 파일 생성 ──
    pipeline_bar.set_description("제출 파일 생성")
    sample_submission = load_sample_submission()
    generate_submission(y_pred_log, sample_submission)
    pipeline_bar.update(1)

    pipeline_bar.set_description("완료")
    pipeline_bar.close()
    print("\n" + "=" * 50)
    print("파이프라인 실행 완료!")
    print(f"OOF RMSE: {cv_results['oof_rmse']:.4f}")
    print(f"평균 Fold RMSE: {cv_results['mean_rmse']:.4f} (+/- {cv_results['std_rmse']:.4f})")
    if args.multi_seed:
        print(f"다중 시드 앙상블: {len(MULTI_SEED_LIST)} seeds")
    print("=" * 50)


def _postprocess_predictions(
    y_pred_log: np.ndarray,
    y_train_log: np.ndarray = None,
) -> np.ndarray:
    """예측값 후처리 — 극단값 클리핑"""
    if y_train_log is not None:
        lower = y_train_log.min()
        upper = y_train_log.max()
        n_clipped = int(np.sum((y_pred_log < lower) | (y_pred_log > upper)))
        y_pred_log = np.clip(y_pred_log, lower, upper)
        if n_clipped > 0:
            tqdm.write(f"  후처리: {n_clipped}건 극단값 클리핑 (범위: [{lower:.4f}, {upper:.4f}])")
    return y_pred_log


def _print_feature_importance(model: LightGBMModel, top_n: int = 20):
    """피처 중요도 상위 N개 출력"""
    if model.model is None:
        return
    try:
        importance = model.model.feature_importances_
        feature_names = model.feature_names
        if importance is None or feature_names is None:
            return

        sorted_idx = np.argsort(importance)[::-1]
        print(f"\n{'='*50}")
        print(f"피처 중요도 (Top {top_n})")
        print(f"{'='*50}")
        for i, idx in enumerate(sorted_idx[:top_n]):
            print(f"  {i+1:2d}. {feature_names[idx]:<30s} {importance[idx]:>8d}")

        # 중요도 0인 피처 수
        zero_importance = int(np.sum(importance == 0))
        if zero_importance > 0:
            print(f"\n  중요도 0인 피처: {zero_importance}개 / 전체 {len(feature_names)}개")
    except Exception:
        pass


if __name__ == "__main__":
    main()
