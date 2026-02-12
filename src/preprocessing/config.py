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
