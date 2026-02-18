"""
전처리 설정 (Configuration)

모든 전처리 단계에서 사용하는 상수와 설정값을 중앙 관리합니다.
팀원이 설정만 변경하면 전체 파이프라인에 반영됩니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_project_root() -> Path:
    """프로젝트 루트 경로를 안정적으로 찾습니다.

    우선순위:
        1. 환경변수 PROJECT_ROOT
        2. 현재 파일 기준 상대 경로 (src/preprocessing/config.py → ../../)
        3. 절대 경로 폴백
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # 현재 파일 기준: src/preprocessing/config.py → house-price-prediction/
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "assets" / "data").exists():
        return candidate

    # CWD 기반 탐색
    cwd = Path(os.getcwd())
    if cwd.name == "notebooks":
        return cwd.parent
    if (cwd / "assets" / "data").exists():
        return cwd

    return cwd


@dataclass
class PreprocessingConfig:
    """전처리 파이프라인의 모든 설정을 담는 데이터 클래스.

    Builder 패턴 스타일로 체이닝 가능:
        config = PreprocessingConfig()
        config.missing_threshold = 0.90
    """

    # ── 경로 설정 ──
    project_root: Path = field(default_factory=_resolve_project_root)

    @property
    def data_dir(self) -> Path:
        remote = Path("/data/ephemeral/home/data")
        if remote.exists():
            return remote
        return self.project_root / "assets" / "data"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "notebooks" / "data"

    # ── 결측값 처리 ──
    missing_threshold: float = 0.80
    coord_preserve_cols: list[str] = field(
        default_factory=lambda: ["좌표X", "좌표Y"]
    )

    # ── 이상치 처리 ──
    outlier_clip_cols: list[str] = field(
        default_factory=lambda: ["전용면적", "층", "건축년도", "주차대수", "k-전체세대수"]
    )
    outlier_iqr_factor: float = 3.0
    target_lower_percentile: float = 0.001
    target_upper_percentile: float = 0.999

    # ── 범주형 식별 ──
    categorical_unique_ratio: float = 0.10
    categorical_max_unique: int = 100

    # ── KNN Imputer ──
    knn_n_neighbors: int = 5
    knn_sample_size: int = 30_000
    knn_chunk_size: int = 50_000

    # ── Float → Int64 변환 대상 ──
    int_convert_cols: list[str] = field(
        default_factory=lambda: [
            "본번", "부번", "k-전체동수", "k-전체세대수", "주차대수",
            "k-연면적", "k-주거전용면적", "k-관리비부과면적",
            "k-전용면적별세대현황(60㎡이하)", "k-전용면적별세대현황(60㎡~85㎡이하)",
            "k-85㎡~135㎡이하", "k-135㎡초과",
        ]
    )

    # ── Target Encoding ──
    target_encode_cols: list[str] = field(
        default_factory=lambda: [
            "아파트명", "도로명", "번지", "시군구", "구", "동",
        ]
    )
    target_encode_smoothing: int = 100

    # ── 버스/지하철 거리 피처 ──
    bus_feature_file: str = "bus_feature.csv"
    subway_feature_file: str = "subway_feature.csv"
    transit_bus_radius_m: float = 500.0
    transit_subway_radius_m: float = 1000.0

    # ── 최신 데이터 필터링 (Exp06) ──
    recent_data_year_from: int | None = 2017

    # ── 공간 클러스터링 (Exp08, Exp10) ──
    spatial_n_clusters: int = 150
    # "kmeans" | "dbscan" | "hdbscan" (Exp10: DBSCAN/HDBSCAN 대안)
    spatial_clustering_method: str = "kmeans"
    # DBSCAN 파라미터 (spatial_clustering_method="dbscan"일 때)
    dbscan_eps: float = 0.05  # 표준화 좌표 기준
    dbscan_min_samples: int = 5
    # HDBSCAN 파라미터 (spatial_clustering_method="hdbscan"일 때)
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int | None = None

    # ── 다중공선성 제거 (PREPROCESSING_PIPELINE Step 3) ──
    multicollinearity_drop_cols: list[str] = field(
        default_factory=lambda: [
            "k-연면적",  # 주거전용면적과 상관계수 0.98 → 설명력 낮은 쪽 제거
        ]
    )

    # ── 주차대수 RF 예측 피처 (PREPROCESSING_PIPELINE Step 10) ──
    parking_prediction_features: list[str] = field(
        default_factory=lambda: [
            "전용면적", "k-전체세대수", "k-전체동수", "건물나이", "층",
        ]
    )

    # ── Label Encoding 대상 (PREPROCESSING_PIPELINE Step 14) ──
    # 빈 리스트면 자동 감지 (object dtype + 고유값 20 미만)
    label_encode_cols: list[str] = field(default_factory=list)

    # ── 저중요도 피처 제거 ──
    low_importance_cols: list[str] = field(
        default_factory=lambda: [
            "관리비 업로드", "k-난방방식", "경비비관리형태",
            "k-단지분류_아파트_주상복합등등_", "사용허가여부",
            "청소비관리형태", "기타/의무/임대/임의_1/2/3/4",
            "세대전기계약방법", "k-관리방식", "k-복도유형",
            "is_real_coord", "k-전체세대수",  # 3모델 모두 importance=0
        ]
    )

    # ── 시계열 피처 (Exp10) ──
    # PREPROCESSING_PIPELINE 21단계에 미포함 — 기본 비활성화 (대규모 데이터에서 O(n²))
    use_timeseries_features: bool = False
    timeseries_group_col: str = "동"  # "동" | "구" | "시군구"

    # ── Adversarial Validation (Exp10) ──
    use_adversarial_validation: bool = True  # Train/Test 구분력 높은 피처 제거 (성능 최적화 전략)
    adversarial_top_n_remove: int = 5  # Train/Test 구분력 높은 상위 N개 피처 제거

    # ── Feature Selection (Exp10) ──
    use_feature_selection: bool = True  # 상위 K개 피처만 유지 (성능 최적화 전략)
    feature_selection_method: str = "permutation"  # "permutation" | "shap"
    feature_selection_top_k: int = 40  # 상위 K개 피처만 유지

    # ── 외부 데이터: 한강 거리 (Exp10) ──
    use_hangang_distance: bool = True  # 좌표 기반 한강까지 거리 근사

    # ── Feature Diet (Exp07) ──
    # 파생 피처로 대체된 원본 컬럼을 제거하여 과적합 방지
    feature_diet_cols: list[str] = field(
        default_factory=lambda: [
            "계약일",
            "주차대수",
        ]
    )

    # ── 원격 서버 프리셋 (GPU/대용량 RAM) ──
    @classmethod
    def remote_server_preset(cls) -> "PreprocessingConfig":
        """원격 GPU 서버(251GB RAM, 64코어)에 최적화된 설정."""
        return cls(
            knn_sample_size=30_000,
            knn_chunk_size=50_000,
        )

    # ── Kakao API ──
    kakao_api_key: str = field(default_factory=lambda: os.environ.get("KAKAO_API_KEY", ""))
    kakao_delay_sec: float = 0.15

    def __post_init__(self) -> None:
        # dotenv 자동 로드 시도
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if not self.kakao_api_key:
                self.kakao_api_key = os.environ.get("KAKAO_API_KEY", "")
        except ImportError:
            pass

        # 원격 서버 자동 감지 (KNN 파라미터는 기본값 유지 — 트리 모델이 결측 자체 처리)
