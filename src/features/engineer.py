"""피처 엔지니어링 함수 (벡터화 연산으로 고속 처리)"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.neighbors import BallTree
from sklearn.model_selection import KFold

from src.config import RANDOM_SEED


# ──────────────────────────────────────────────
# 좌표 컬럼 자동 탐지 유틸리티
# ──────────────────────────────────────────────


def _find_coord_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """DataFrame에서 좌표 컬럼을 자동 탐지"""
    possible_x = ["좌표X", "좌표x", "경도", "X좌표", "X", "x", "lon", "longitude"]
    possible_y = ["좌표Y", "좌표y", "위도", "Y좌표", "Y", "y", "lat", "latitude"]

    x_col = next((c for c in possible_x if c in df.columns), None)
    y_col = next((c for c in possible_y if c in df.columns), None)
    return x_col, y_col


# ──────────────────────────────────────────────
# BallTree 기반 고속 거리·개수 계산
# ──────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0


def _build_ball_tree(lat: np.ndarray, lon: np.ndarray) -> BallTree:
    coords_rad = np.deg2rad(np.column_stack([lat, lon]))
    return BallTree(coords_rad, metric="haversine")


def _nearest_distance_km(
    query_lat: np.ndarray, query_lon: np.ndarray, tree: BallTree,
) -> np.ndarray:
    query_rad = np.deg2rad(np.column_stack([query_lat, query_lon]))
    dist_rad, _ = tree.query(query_rad, k=1)
    return dist_rad.ravel() * EARTH_RADIUS_KM


def _count_within_radius_km(
    query_lat: np.ndarray, query_lon: np.ndarray, tree: BallTree, radius_km: float,
) -> np.ndarray:
    query_rad = np.deg2rad(np.column_stack([query_lat, query_lon]))
    radius_rad = radius_km / EARTH_RADIUS_KM
    counts = tree.query_radius(query_rad, r=radius_rad, count_only=True)
    return counts.astype(int)


# ══════════════════════════════════════════════
# K-Fold Target Encoding
# ══════════════════════════════════════════════


class KFoldTargetEncoder:
    """고카디널리티 범주형 피처의 K-Fold Target Encoding

    학습 시: K-Fold 기반으로 데이터 누수를 방지하면서 범주형 피처를
            target 값의 평활(smoothed) 평균으로 인코딩합니다.
    테스트 시: 전체 학습 데이터 기반 통계를 사용합니다.
    """

    DEFAULT_COLS = [
        "아파트명", "시군구", "구군", "동",
        "k-관리방식", "k-난방방식", "k-복도유형",
        "k-건설사_시공사", "k-세대타입_분양형태", "k-단지분류_아파트_주상복합등등",
    ]

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        n_folds: int = 5,
        smoothing: float = 20.0,
        random_state: int = RANDOM_SEED,
    ):
        self.cols = cols if cols is not None else self.DEFAULT_COLS
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean: Optional[float] = None
        self.encoding_maps: Dict[str, pd.Series] = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """학습 시: K-Fold 기반 target encoding (데이터 누수 방지)"""
        X = X.copy()
        self.global_mean = float(y.mean())
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        active_cols = [c for c in self.cols if c in X.columns]

        tqdm.write(
            f"  Target Encoding: {len(active_cols)}개 컬럼 "
            f"({self.n_folds}-Fold, smoothing={self.smoothing})"
        )

        for col in active_cols:
            encoded = pd.Series(np.nan, index=X.index, dtype=float)
            col_str = X[col].astype(str)

            for train_idx, val_idx in kf.split(X):
                tmp = pd.DataFrame({
                    "target": y.iloc[train_idx].values,
                    "cat": col_str.iloc[train_idx].values,
                })
                stats = tmp.groupby("cat")["target"].agg(["mean", "count"])
                smoothed = (
                    stats["mean"] * stats["count"]
                    + self.global_mean * self.smoothing
                ) / (stats["count"] + self.smoothing)
                mapped = col_str.iloc[val_idx].map(smoothed)
                encoded.iloc[val_idx] = mapped.values

            encoded = encoded.fillna(self.global_mean)
            X[f"{col}_te"] = encoded

            # 전체 학습 데이터 기반 매핑 저장 (테스트용)
            full_tmp = pd.DataFrame({
                "target": y.values,
                "cat": col_str.values,
            })
            full_stats = full_tmp.groupby("cat")["target"].agg(["mean", "count"])
            self.encoding_maps[col] = (
                full_stats["mean"] * full_stats["count"]
                + self.global_mean * self.smoothing
            ) / (full_stats["count"] + self.smoothing)

        tqdm.write(f"  Target Encoding 완료: +{len(active_cols)}개 피처 생성")
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """테스트 시: 전체 학습 데이터 통계 사용"""
        X = X.copy()
        for col, mapping in self.encoding_maps.items():
            if col not in X.columns:
                continue
            col_str = X[col].astype(str)
            X[f"{col}_te"] = col_str.map(mapping).fillna(self.global_mean)
        return X

    def save(self, filepath: Path):
        """인코더 상태 저장"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "cols": self.cols,
                "global_mean": self.global_mean,
                "encoding_maps": self.encoding_maps,
                "smoothing": self.smoothing,
            }, f)

    def load(self, filepath: Path):
        """인코더 상태 로드"""
        with open(Path(filepath), "rb") as f:
            data = pickle.load(f)
        self.cols = data["cols"]
        self.global_mean = data["global_mean"]
        self.encoding_maps = data["encoding_maps"]
        self.smoothing = data["smoothing"]


# ══════════════════════════════════════════════
# 지역 피처 (시군구 파싱)
# ══════════════════════════════════════════════


def add_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """시군구를 시도/구군/동으로 분리"""
    df = df.copy()
    if "시군구" not in df.columns:
        return df

    parts = df["시군구"].astype(str).str.split(expand=True)
    df["시도"] = parts[0] if 0 in parts.columns else "미상"
    df["구군"] = parts[1] if 1 in parts.columns else "미상"
    df["동"] = parts[2] if 2 in parts.columns else "미상"

    for col in ["시도", "구군", "동"]:
        df[col] = df[col].fillna("미상")

    return df


# ──────────────────────────────────────────────
# 버스 / 지하철 피처
# ──────────────────────────────────────────────


def add_bus_features(df: pd.DataFrame, bus_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    x_col, y_col = _find_coord_cols(df)
    if x_col is None or y_col is None:
        tqdm.write("  경고: 좌표 컬럼 없음 → 버스 피처 건너뜀")
        df["bus_nearest_dist_m"] = np.nan
        df["bus_count_500m"] = 0
        return df

    bus_x = "X좌표" if "X좌표" in bus_df.columns else bus_df.columns[3]
    bus_y = "Y좌표" if "Y좌표" in bus_df.columns else bus_df.columns[4]

    bus_clean = bus_df[[bus_x, bus_y]].dropna()
    bus_lon = bus_clean[bus_x].values.astype(float)
    bus_lat = bus_clean[bus_y].values.astype(float)

    apt_lon = df[x_col].values.astype(float)
    apt_lat = df[y_col].values.astype(float)
    valid = ~(np.isnan(apt_lon) | np.isnan(apt_lat))

    tqdm.write(f"  버스: 유효 좌표 {valid.sum():,}건 / 정류소 {len(bus_lat):,}개")

    tree = _build_ball_tree(bus_lat, bus_lon)

    dist_m = np.full(len(df), np.nan)
    dist_m[valid] = _nearest_distance_km(apt_lat[valid], apt_lon[valid], tree) * 1000

    count_500 = np.zeros(len(df), dtype=int)
    count_500[valid] = _count_within_radius_km(apt_lat[valid], apt_lon[valid], tree, radius_km=0.5)

    df["bus_nearest_dist_m"] = dist_m
    df["bus_count_500m"] = count_500
    return df


def add_subway_features(df: pd.DataFrame, subway_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    x_col, y_col = _find_coord_cols(df)
    if x_col is None or y_col is None:
        tqdm.write("  경고: 좌표 컬럼 없음 → 지하철 피처 건너뜀")
        df["subway_nearest_dist_m"] = np.nan
        df["subway_count_1km"] = 0
        return df

    sub_lat_col = "위도" if "위도" in subway_df.columns else subway_df.columns[3]
    sub_lon_col = "경도" if "경도" in subway_df.columns else subway_df.columns[4]

    sub_clean = subway_df[[sub_lat_col, sub_lon_col]].dropna()
    sub_lat = sub_clean[sub_lat_col].values.astype(float)
    sub_lon = sub_clean[sub_lon_col].values.astype(float)

    apt_lon = df[x_col].values.astype(float)
    apt_lat = df[y_col].values.astype(float)
    valid = ~(np.isnan(apt_lon) | np.isnan(apt_lat))

    tqdm.write(f"  지하철: 유효 좌표 {valid.sum():,}건 / 역 {len(sub_lat):,}개")

    tree = _build_ball_tree(sub_lat, sub_lon)

    dist_m = np.full(len(df), np.nan)
    dist_m[valid] = _nearest_distance_km(apt_lat[valid], apt_lon[valid], tree) * 1000

    count_1k = np.zeros(len(df), dtype=int)
    count_1k[valid] = _count_within_radius_km(apt_lat[valid], apt_lon[valid], tree, radius_km=1.0)

    df["subway_nearest_dist_m"] = dist_m
    df["subway_count_1km"] = count_1k
    return df


# ──────────────────────────────────────────────
# 시간 / 건축년도 / 단지 피처
# ──────────────────────────────────────────────


def add_time_features(df: pd.DataFrame, date_col: str = "계약년월") -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        return df

    ym = df[date_col].astype(str)
    df["contract_year"] = ym.str[:4].astype(int)
    df["contract_month"] = ym.str[4:6].astype(int)
    df["contract_quarter"] = ((df["contract_month"] - 1) // 3) + 1
    df["contract_ym"] = df[date_col].astype(int) if df[date_col].dtype != object else ym.astype(int)

    # 월 순환 인코딩 (계절성 캡처)
    df["month_sin"] = np.sin(2 * np.pi * df["contract_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["contract_month"] / 12)

    return df


def add_building_age_features(
    df: pd.DataFrame, build_year_col: str = "건축년도", contract_year_col: str = "contract_year",
) -> pd.DataFrame:
    df = df.copy()
    if build_year_col not in df.columns:
        return df

    valid = df[build_year_col].notna() & (df[build_year_col] > 0)
    if contract_year_col in df.columns:
        df["building_age"] = np.where(valid, df[contract_year_col] - df[build_year_col], np.nan)
    else:
        df["building_age"] = np.where(valid, 2024 - df[build_year_col], np.nan)
    return df


def add_complex_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "주차대수" in df.columns and "k-전체세대수" in df.columns:
        df["parking_per_household"] = df["주차대수"] / (df["k-전체세대수"] + 1e-6)
    return df


# ══════════════════════════════════════════════
# 교호작용 피처
# ══════════════════════════════════════════════


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """면적, 층, 건축년수 등의 교호작용 피처 생성"""
    df = df.copy()

    # 면적 구간 (한국 아파트 규모 기준 고정 bin)
    if "전용면적" in df.columns:
        area_bins = [0, 30, 40, 50, 60, 85, 100, 120, 150, 200, 9999]
        df["area_bin"] = pd.cut(
            df["전용면적"].fillna(0), bins=area_bins, labels=False,
        ).fillna(0).astype(int)

    # 층 그룹 (저층/중층/고층/초고층)
    if "층" in df.columns:
        df["floor_group"] = pd.cut(
            df["층"].fillna(0),
            bins=[-999, 3, 10, 20, 999],
            labels=False,
        ).fillna(0).astype(int)

    # 면적 x 층
    if "전용면적" in df.columns and "층" in df.columns:
        df["area_x_floor"] = df["전용면적"].fillna(0) * df["층"].fillna(0)

    # 면적 x 건축년수
    if "전용면적" in df.columns and "building_age" in df.columns:
        df["area_x_age"] = df["전용면적"].fillna(0) * df["building_age"].fillna(0)

    return df


# ══════════════════════════════════════════════
# 지역별 집계 통계 (빈도·분산 인코딩)
# ══════════════════════════════════════════════


def compute_region_stats(
    X: pd.DataFrame, y: pd.Series,
) -> Dict[str, Dict[str, pd.Series]]:
    """구/동/시군구별 빈도 및 가격 통계 계산 (학습 데이터 기반)"""
    stats: Dict[str, Dict[str, pd.Series]] = {}

    for col in ["구군", "동", "시군구"]:
        if col not in X.columns:
            continue
        col_str = X[col].astype(str)
        tmp = pd.DataFrame({"target": y.values, "cat": col_str.values})
        group = tmp.groupby("cat")["target"]
        stats[col] = {
            "freq": group.count(),
            "price_std": group.std().fillna(0),
            "price_median": group.median(),
        }
        tqdm.write(f"  지역 통계: {col} — {len(stats[col]['freq'])}개 고유값")

    return stats


def apply_region_stats(
    X: pd.DataFrame,
    stats: Dict[str, Dict[str, pd.Series]],
) -> pd.DataFrame:
    """미리 계산된 지역 통계를 적용"""
    X = X.copy()
    for col, stat_dict in stats.items():
        if col not in X.columns:
            continue
        col_str = X[col].astype(str)
        X[f"{col}_freq"] = col_str.map(stat_dict["freq"]).fillna(0).astype(int)
        X[f"{col}_price_std"] = col_str.map(stat_dict["price_std"]).fillna(0)
        X[f"{col}_price_median"] = col_str.map(stat_dict["price_median"]).fillna(0)
    return X


def save_region_stats(stats: Dict, filepath: Path):
    """지역 통계 저장"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(stats, f)


def load_region_stats(filepath: Path) -> Dict:
    """지역 통계 로드"""
    with open(Path(filepath), "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════
# 시간 기반 통계 (아파트/지역별 가격 추세 + 거래 빈도)
# ══════════════════════════════════════════════


def compute_time_lag_stats(
    X: pd.DataFrame, y: pd.Series,
) -> Dict[str, pd.Series]:
    """아파트/지역별 시간 기반 통계 계산 (학습 데이터 기반)

    - 거래 건수 (빈도 인코딩)
    - 최근 2년 평균가
    - 가격 추세 (최근 2년 평균 / 전체 평균 - 1)
    """
    stats: Dict[str, pd.Series] = {}

    if "contract_ym" not in X.columns:
        tqdm.write("  경고: contract_ym 컬럼 없음 → 시간 통계 건너뜀")
        return stats

    max_ym = int(X["contract_ym"].max())
    recent_cutoff = max_ym - 200  # ~2년 전

    base_df = pd.DataFrame({
        "target": y.values,
        "contract_ym": X["contract_ym"].values,
    })

    for col in ["아파트명", "동", "구군"]:
        if col not in X.columns:
            continue
        base_df["cat"] = X[col].astype(str).values

        # 전체 거래 건수
        stats[f"{col}_tx_count"] = base_df.groupby("cat")["target"].count()

        # 전체 평균가
        full_mean = base_df.groupby("cat")["target"].mean()

        # 최근 2년 평균가
        recent_mask = base_df["contract_ym"] >= recent_cutoff
        if recent_mask.sum() > 0:
            recent_df = base_df[recent_mask]
            recent_mean = recent_df.groupby("cat")["target"].mean()
            stats[f"{col}_recent_price"] = recent_mean

            # 가격 추세 (최근/전체 - 1)
            trend = (recent_mean / full_mean).reindex(full_mean.index).fillna(0.0) - 1.0
            stats[f"{col}_price_trend"] = trend

        tqdm.write(
            f"  시간 통계: {col} — {len(stats.get(f'{col}_tx_count', []))}개 고유값, "
            f"최근 기준={recent_cutoff}"
        )

    return stats


def apply_time_lag_stats(
    X: pd.DataFrame, stats: Dict[str, pd.Series],
) -> pd.DataFrame:
    """시간 기반 통계를 적용"""
    X = X.copy()
    applied = 0
    for key, values in stats.items():
        for base_col in ["아파트명", "동", "구군"]:
            if key.startswith(f"{base_col}_"):
                if base_col in X.columns:
                    col_str = X[base_col].astype(str)
                    X[key] = col_str.map(values).fillna(0)
                    applied += 1
                break
    tqdm.write(f"  시간 통계 적용: +{applied}개 피처")
    return X


def save_time_lag_stats(stats: Dict, filepath: Path):
    """시간 기반 통계 저장"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(stats, f)


def load_time_lag_stats(filepath: Path) -> Dict:
    """시간 기반 통계 로드"""
    with open(Path(filepath), "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════
# 메인 엔트리포인트
# ══════════════════════════════════════════════


def engineer_features(
    df: pd.DataFrame,
    bus_df: pd.DataFrame = None,
    subway_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """모든 기본 피처 엔지니어링 적용 (target 불필요)

    Target Encoding과 지역 통계는 target 변수가 필요하므로
    별도 함수(KFoldTargetEncoder, compute_region_stats)로 처리합니다.
    """
    df = df.copy()
    original_cols = len(df.columns)

    steps = ["지역 파싱", "시간 피처", "건축년도 피처"]
    if bus_df is not None:
        steps.append("버스 피처")
    if subway_df is not None:
        steps.append("지하철 피처")
    steps += ["단지 피처", "교호작용 피처"]

    bar = tqdm(steps, desc="피처 엔지니어링", unit="단계", leave=True, colour="magenta")

    # 지역 파싱 (시군구 → 시도/구군/동)
    bar.set_description("피처 엔지니어링 - 지역 파싱")
    df = add_region_features(df)
    bar.update(1)

    # 시간 피처
    bar.set_description("피처 엔지니어링 - 시간")
    df = add_time_features(df)
    bar.update(1)

    # 건축년도 피처
    bar.set_description("피처 엔지니어링 - 건축년도")
    df = add_building_age_features(df)
    bar.update(1)

    # 버스 피처
    if bus_df is not None:
        bar.set_description("피처 엔지니어링 - 버스")
        df = add_bus_features(df, bus_df)
        bar.update(1)

    # 지하철 피처
    if subway_df is not None:
        bar.set_description("피처 엔지니어링 - 지하철")
        df = add_subway_features(df, subway_df)
        bar.update(1)

    # 단지 규모 피처
    bar.set_description("피처 엔지니어링 - 단지")
    df = add_complex_features(df)
    bar.update(1)

    # 교호작용 피처
    bar.set_description("피처 엔지니어링 - 교호작용")
    df = add_interaction_features(df)
    bar.update(1)

    bar.set_description("피처 엔지니어링 완료")
    bar.close()

    new_cols = len(df.columns) - original_cols
    tqdm.write(f"  추가된 피처: {new_cols}개 / 최종 피처: {len(df.columns)}개")

    return df
