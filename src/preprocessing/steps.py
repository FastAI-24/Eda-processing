"""
전처리 단계 구현 (Concrete Strategy Classes)

각 단계는 PreprocessingStep을 상속하며 독립적으로 테스트/교체 가능합니다.

파이프라인 순서:
    Step 0   → 취소 거래 필터링
    Step 1   → Target 분리
    Step 2   → 고결측 컬럼 제거
    Step 2.5 → Float→Int64 타입 변환
    Step 3   → 컬럼명 정리 (LightGBM 호환)
    Step 3.5 → 날짜/주소 파생 피처
    Step 4   → 범주형 컬럼 식별
    Step 5   → 결측 지표 피처
    Step 6   → 좌표 보간 (Kakao API + 시군구 평균)
    Step 7   → 결측값 대체 (KNN Imputer)
    Step 7.5 → 세대당 주차대수 파생 피처
    Step 8   → 이상치 클리핑
    Step 9   → Target 로그 변환
"""

from __future__ import annotations

import json as _json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from tqdm.auto import tqdm

from .base import PreprocessingContext, PreprocessingStep


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 0: 취소 거래 필터링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FilterCancelledTransactionsStep(PreprocessingStep):
    """해제사유발생일이 존재하는 행(취소 거래)을 제거합니다.

    취소된 거래는 실제 유효한 거래가 아니므로, 학습 전에 제거하여
    모델이 잘못된 패턴을 학습하지 않도록 합니다.
    단, 테스트 데이터는 행을 삭제하면 안 되므로 컬럼만 제거합니다.
    """

    @property
    def name(self) -> str:
        return "Step 0: 취소 거래 필터링"

    @staticmethod
    def _filter(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """취소 거래를 필터링합니다.

        Args:
            df: 대상 DataFrame
            is_train: True면 취소 거래 행 제거, False면 컬럼만 제거
        """
        col = "해제사유발생일"
        if col in df.columns:
            n_cancelled = df[col].notnull().sum()
            if n_cancelled > 0:
                print(f"  취소 거래 {n_cancelled:,}건 발견")
                if is_train:
                    print("  -> 학습 데이터: 취소 거래 행 제거")
                    df = df[df[col].isnull()].copy()
                else:
                    print("  -> 테스트 데이터: 행 유지 (삭제 안 함)")
            else:
                print("  취소 거래 없음")

            # 컬럼은 학습/테스트 모두 제거
            df = df.drop(columns=[col], errors="ignore")
        else:
            print(f"  '{col}' 컬럼 없음 — 건너뜀")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        print(f"Before: train={ctx.raw_train_df.shape}, test={ctx.raw_test_df.shape}")
        # Train: 행 제거 O
        ctx.raw_train_df = self._filter(ctx.raw_train_df, is_train=True)
        # Test: 행 제거 X (컬럼만 제거)
        ctx.raw_test_df = self._filter(ctx.raw_test_df, is_train=False)
        print(f"After:  train={ctx.raw_train_df.shape}, test={ctx.raw_test_df.shape}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Target 분리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TargetSeparationStep(PreprocessingStep):
    """학습 데이터에서 target 컬럼을 분리합니다."""

    @property
    def name(self) -> str:
        return "Step 1: Target 분리"

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        ctx.y_train = ctx.raw_train_df["target"].copy()
        ctx.X_train = ctx.raw_train_df.drop(columns=["target"]).copy()
        ctx.X_test = ctx.raw_test_df.copy()

        print(f"  X_train: {ctx.X_train.shape}")
        print(f"  y_train: {ctx.y_train.shape}")
        print(f"  X_test:  {ctx.X_test.shape}")
        print(f"\n  target 기본 통계:")
        print(f"  {ctx.y_train.describe().to_string()}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: 고결측 컬럼 제거
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RemoveHighMissingColumnsStep(PreprocessingStep):
    """결측 비율이 threshold 이상인 컬럼을 제거합니다."""

    @property
    def name(self) -> str:
        return "Step 2: 고결측 컬럼 제거"

    @staticmethod
    def _remove(df: pd.DataFrame, threshold: float, preserve_cols: list[str]) -> pd.DataFrame:
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio >= threshold].index.tolist()
        cols_to_drop = [c for c in cols_to_drop if c not in preserve_cols]
        print(f"  제거 대상 ({len(cols_to_drop)}개): {cols_to_drop}")
        return df.drop(columns=cols_to_drop, errors="ignore")

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cfg = ctx.config
        print(f"  Before: {ctx.X_train.shape}")
        ctx.X_train = self._remove(ctx.X_train, cfg.missing_threshold, cfg.coord_preserve_cols)
        ctx.X_test = self._remove(ctx.X_test, cfg.missing_threshold, cfg.coord_preserve_cols)
        print(f"  After: {ctx.X_train.shape}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2.5: Float → Int64 타입 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FloatToIntConversionStep(PreprocessingStep):
    """정수형 데이터를 가진 Float 컬럼을 Nullable Int64로 변환합니다."""

    @property
    def name(self) -> str:
        return "Step 2.5: Float→Int64 타입 변환"

    @staticmethod
    def _convert(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        converted = []
        for col in cols:
            if col in df.columns and df[col].dtype in ("float64", "float32"):
                try:
                    df[col] = df[col].round().astype("Int64")
                    converted.append(col)
                except Exception:
                    pass
        print(f"  Int64 변환 완료 ({len(converted)}개): {converted}")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cols = ctx.config.int_convert_cols
        print("  학습 데이터:")
        ctx.X_train = self._convert(ctx.X_train, cols)
        print("  테스트 데이터:")
        ctx.X_test = self._convert(ctx.X_test, cols)
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: 컬럼명 정리 (LightGBM 호환)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SanitizeColumnNamesStep(PreprocessingStep):
    """LightGBM 호환을 위해 컬럼명의 특수문자를 밑줄로 치환합니다."""

    @property
    def name(self) -> str:
        return "Step 3: 컬럼명 정리 (LightGBM 호환)"

    @staticmethod
    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for col in df.columns:
            new_col = re.sub(r'[{}\[\]"\\,()=]', '_', col)
            if new_col != col:
                rename_map[col] = new_col
        if rename_map:
            print(f"  변경된 컬럼명 ({len(rename_map)}개):")
            for old, new in list(rename_map.items())[:10]:
                print(f"    {old} → {new}")
            if len(rename_map) > 10:
                print(f"    ... 외 {len(rename_map) - 10}개")
            df = df.rename(columns=rename_map)
        else:
            print("  변경할 컬럼명 없음")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        ctx.X_train = self._sanitize(ctx.X_train)
        ctx.X_test = self._sanitize(ctx.X_test)
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3.5: 날짜/주소 파생 피처
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DateAddressFeaturesStep(PreprocessingStep):
    """날짜 파싱, 주소 분리, 경과연수 등 파생 피처를 생성합니다."""

    @property
    def name(self) -> str:
        return "Step 3.5: 날짜/주소 파생 피처"

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        created = []

        # 1) 계약일자: 계약년월(202301) + 계약일(15) → 20230115 (정수)
        if "계약년월" in df.columns and "계약일" in df.columns:
            df["계약일자"] = (
                df["계약년월"].astype(str)
                + df["계약일"].astype(str).str.zfill(2)
            ).astype("Int64")
            created.append("계약일자")

        # 2) 시군구 → 구, 동 분리
        if "시군구" in df.columns:
            split = df["시군구"].str.split(" ", expand=True)
            if split.shape[1] >= 2:
                df["구"] = split[1]
                created.append("구")
            if split.shape[1] >= 3:
                df["동"] = split[2]
                created.append("동")

        # 3) 건물나이 (경과연수)
        if "건축년도" in df.columns and "계약년월" in df.columns:
            contract_year = df["계약년월"] // 100
            df["건물나이"] = contract_year - df["건축년도"]
            df["건물나이"] = df["건물나이"].clip(lower=0)
            created.append("건물나이")

        print(f"  생성된 파생 피처 ({len(created)}개): {created}")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        print("  학습 데이터:")
        ctx.X_train = self._add_features(ctx.X_train)
        print("  테스트 데이터:")
        ctx.X_test = self._add_features(ctx.X_test)
        print(f"  현재 컬럼 수: X_train={ctx.X_train.shape[1]}, X_test={ctx.X_test.shape[1]}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3.7: 시간 파생 피처
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TemporalFeaturesStep(PreprocessingStep):
    """계약년월에서 세부 시간 피처를 분해하여 계절성/추세를 포착합니다.

    생성 피처:
        - 계약년: 연도 (2007~2023)
        - 계약월: 월 (1~12)
        - 계약분기: 분기 (1~4)
        - 계약반기: 반기 (1~2)
        - 계약월_sin, 계약월_cos: 월의 순환 인코딩 (12월→1월 연속성)
    """

    @property
    def name(self) -> str:
        return "Step 3.7: 시간 파생 피처"

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        created: list[str] = []

        if "계약년월" in df.columns:
            df["계약년"] = (df["계약년월"] // 100).astype("Int64")
            df["계약월"] = (df["계약년월"] % 100).astype("Int64")
            df["계약분기"] = ((df["계약월"] - 1) // 3 + 1).astype("Int64")
            df["계약반기"] = ((df["계약월"] - 1) // 6 + 1).astype("Int64")

            # Cyclical encoding: 12월 → 1월 연속성 유지
            month_float = df["계약월"].astype("float64")
            df["계약월_sin"] = np.sin(2 * np.pi * month_float / 12)
            df["계약월_cos"] = np.cos(2 * np.pi * month_float / 12)

            created.extend(["계약년", "계약월", "계약분기", "계약반기", "계약월_sin", "계약월_cos"])

        print(f"  생성된 시간 피처 ({len(created)}개): {created}")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        print("  학습 데이터:")
        ctx.X_train = self._add_features(ctx.X_train)
        print("  테스트 데이터:")
        ctx.X_test = self._add_features(ctx.X_test)
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: 범주형 컬럼 식별
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class IdentifyCategoricalColumnsStep(PreprocessingStep):
    """범주형 컬럼을 자동 식별합니다."""

    @property
    def name(self) -> str:
        return "Step 4: 범주형 컬럼 식별"

    @staticmethod
    def _identify(
        df: pd.DataFrame,
        unique_ratio: float,
        max_unique: int,
    ) -> list[str]:
        categorical_cols: list[str] = []
        n = len(df)
        for col in df.columns:
            if df[col].dtype == object:
                categorical_cols.append(col)
                continue
            nunique = df[col].nunique()
            if n > 0 and (nunique / n) < unique_ratio and nunique < max_unique:
                categorical_cols.append(col)
        return categorical_cols

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cfg = ctx.config
        ctx.categorical_cols = self._identify(
            ctx.X_train, cfg.categorical_unique_ratio, cfg.categorical_max_unique,
        )
        print(f"  식별된 범주형 컬럼 ({len(ctx.categorical_cols)}개):")
        for i, col in enumerate(ctx.categorical_cols, 1):
            nunique = ctx.X_train[col].nunique()
            dtype = ctx.X_train[col].dtype
            print(f"    {i:2d}. {str(col):<35s} | dtype={str(dtype):<10s} | 고유값={int(nunique):,}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: 결측 지표 피처
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MissingIndicatorStep(PreprocessingStep):
    """각 행의 결측값 총 개수를 새로운 피처로 생성합니다."""

    @property
    def name(self) -> str:
        return "Step 5: 결측 지표 피처"

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        ctx.X_train["missing_count"] = ctx.X_train.isnull().sum(axis=1)
        ctx.X_test["missing_count"] = ctx.X_test.isnull().sum(axis=1)

        print(f"  missing_count 분포 (학습 데이터):")
        print(f"  {ctx.X_train['missing_count'].value_counts().sort_index().head(20).to_string()}")
        print(f"  평균 결측 수: {ctx.X_train['missing_count'].mean():.2f}")
        print(f"  최대 결측 수: {ctx.X_train['missing_count'].max()}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 6: 좌표 보간 (Kakao API + 시군구 평균)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CoordinateInterpolationStep(PreprocessingStep):
    """좌표 결측을 Kakao API → 시군구 평균 → 전체 평균 순으로 보간합니다."""

    @property
    def name(self) -> str:
        return "Step 6: 좌표 보간"

    # ── Kakao API 헬퍼 ──
    @staticmethod
    def _build_address(row: pd.Series) -> str:
        parts = []
        if pd.notna(row.get("시군구")) and str(row["시군구"]).strip():
            parts.append(str(row["시군구"]).strip())
        if pd.notna(row.get("도로명")) and str(row["도로명"]).strip():
            parts.append(str(row["도로명"]).strip())
        elif pd.notna(row.get("번지")) and str(row["번지"]).strip():
            parts.append(str(row["번지"]).strip())
        elif pd.notna(row.get("본번")):
            try:
                b = str(int(row["본번"]))
                s = str(int(row["부번"])) if pd.notna(row.get("부번")) else ""
                parts.append(f"{b}-{s}" if s else b)
            except (ValueError, TypeError):
                pass
        if pd.notna(row.get("아파트명")) and str(row["아파트명"]).strip():
            parts.append(str(row["아파트명"]).strip())
        return " ".join(parts) if parts else ""

    @staticmethod
    def _fetch_coords_from_kakao(
        address: str, api_key: str, cache: dict, delay_sec: float = 0.15,
    ) -> tuple[float, float] | None:
        if not address or not api_key.strip():
            return None
        address = address.strip()
        if address in cache:
            return cache[address]
        try:
            url = (
                "https://dapi.kakao.com/v2/local/search/address.json?"
                + urllib.parse.urlencode({"query": address})
            )
            req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {api_key}"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode())
            docs = data.get("documents", [])
            if docs and docs[0].get("x") and docs[0].get("y"):
                x, y = float(docs[0]["x"]), float(docs[0]["y"])
                cache[address] = (x, y)
                return (x, y)
        except Exception:
            pass
        cache[address] = None
        time.sleep(delay_sec)
        return None

    # ── 핵심 보간 로직 ──
    @classmethod
    def _interpolate(
        cls,
        df: pd.DataFrame,
        region_col: str = "시군구",
        coord_cols: list[str] | None = None,
        stats: dict[str, Any] | None = None,
        use_kakao_api: bool = True,
        kakao_api_key: str = "",
        kakao_delay_sec: float = 0.15,
    ) -> tuple[pd.DataFrame, dict, dict]:
        if coord_cols is None:
            coord_cols = ["좌표X", "좌표Y"]

        df = df.copy()
        coord_group_means: dict[str, dict] = {}
        coord_global_means: dict[str, float] = {}
        kakao_cache: dict | None = {} if use_kakao_api and kakao_api_key else None

        # 1단계: Kakao API
        if kakao_cache is not None and "좌표X" in df.columns and "좌표Y" in df.columns:
            missing_both = df["좌표X"].isna() & df["좌표Y"].isna()
            if missing_both.any():
                addr_series = df.loc[missing_both].apply(cls._build_address, axis=1)
                unique_addr = addr_series[addr_series.str.len() > 0].unique()
                addr_to_coord: dict[str, tuple[float, float]] = {}
                for addr in tqdm(unique_addr, desc="Kakao API 주소→좌표 조회", leave=False):
                    r = cls._fetch_coords_from_kakao(addr, kakao_api_key, kakao_cache, 0)
                    if r:
                        addr_to_coord[addr] = r
                    time.sleep(kakao_delay_sec)
                coords = addr_series.map(addr_to_coord)
                valid_idx = coords.dropna().index
                if len(valid_idx) > 0:
                    df.loc[valid_idx, "좌표X"] = coords.loc[valid_idx].apply(lambda c: c[0])
                    df.loc[valid_idx, "좌표Y"] = coords.loc[valid_idx].apply(lambda c: c[1])
                    print(f"  Kakao API: {len(valid_idx):,}건 보간")

        # 2/3단계: 시군구 평균 → 전체 평균
        for coord in coord_cols:
            if coord not in df.columns:
                continue
            before_na = df[coord].isna().sum()

            if stats is not None:
                group_mean = stats.get("coord_group_means", {}).get(coord, {})
                global_mean = stats.get("coord_global_means", {}).get(coord, df[coord].mean())
                if region_col in df.columns and group_mean:
                    df[coord] = df[coord].fillna(df[region_col].map(group_mean))
                df[coord] = df[coord].fillna(global_mean)
            else:
                global_mean = df[coord].mean()
                coord_global_means[coord] = global_mean
                if region_col in df.columns:
                    group_means = df.groupby(region_col)[coord].mean()
                    coord_group_means[coord] = group_means.to_dict()
                    df[coord] = df[coord].fillna(df[region_col].map(group_means))
                df[coord] = df[coord].fillna(global_mean)

            after_na = df[coord].isna().sum()
            print(f"  {coord}: 결측 {before_na:,} → {after_na:,} (보간: {before_na - after_na:,}건)")

        return df, coord_group_means, coord_global_means

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cfg = ctx.config

        # 학습 데이터
        print("  학습 데이터 좌표 보간:")
        ctx.X_train, coord_group_means, coord_global_means = self._interpolate(
            ctx.X_train,
            kakao_api_key=cfg.kakao_api_key,
            kakao_delay_sec=cfg.kakao_delay_sec,
        )
        ctx.train_stats["coord_group_means"] = coord_group_means
        ctx.train_stats["coord_global_means"] = coord_global_means

        # 테스트 데이터 (학습 기준 통계)
        print("\n  테스트 데이터 좌표 보간 (학습 기준 통계 사용):")
        ctx.X_test, _, _ = self._interpolate(
            ctx.X_test, stats=ctx.train_stats,
        )
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 6.5: 공간 파생 피처 (좌표 → 거리)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SpatialFeaturesStep(PreprocessingStep):
    """좌표 기반 공간 피처를 생성합니다.

    보간된 좌표(경도/위도)로부터 서울 주요 랜드마크까지의
    유클리드 거리를 계산하여 지리적 가격 구배를 포착합니다.

    생성 피처:
        - dist_강남역: 강남역까지 거리 (km)
        - dist_서울시청: 서울시청까지 거리 (km)
        - dist_여의도: 여의도까지 거리 (km)
    """

    # 주요 랜드마크 좌표 (경도, 위도)
    LANDMARKS: dict[str, tuple[float, float]] = {
        "강남역": (127.0276, 37.4979),
        "서울시청": (126.9780, 37.5665),
        "여의도": (126.9246, 37.5219),
    }

    @property
    def name(self) -> str:
        return "Step 6.5: 공간 파생 피처"

    @staticmethod
    def _approx_km_distance(
        lon1: pd.Series, lat1: pd.Series,
        lon2: float, lat2: float,
    ) -> pd.Series:
        """서울 지역(위도 ~37.5°) 근사 유클리드 거리 (km).

        위도 1° ≈ 111 km, 경도 1° ≈ 88 km (위도 37.5° 기준).
        """
        dx = (lon1 - lon2) * 88.0
        dy = (lat1 - lat2) * 111.0
        return np.sqrt(dx ** 2 + dy ** 2)

    @classmethod
    def _add_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        created: list[str] = []

        if "좌표X" in df.columns and "좌표Y" in df.columns:
            lon = df["좌표X"].astype("float64")
            lat = df["좌표Y"].astype("float64")

            for name, (ref_lon, ref_lat) in cls.LANDMARKS.items():
                col_name = f"dist_{name}"
                df[col_name] = cls._approx_km_distance(lon, lat, ref_lon, ref_lat)
                created.append(col_name)
                stats = df[col_name].describe()
                print(f"    {col_name}: mean={stats['mean']:.2f}km, max={stats['max']:.2f}km")

        print(f"  생성된 공간 피처 ({len(created)}개): {created}")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        print("  학습 데이터:")
        ctx.X_train = self._add_features(ctx.X_train)
        print("  테스트 데이터:")
        ctx.X_test = self._add_features(ctx.X_test)
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 6.7: 버스/지하철 거리 피처
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TransitFeaturesStep(PreprocessingStep):
    """버스 정류소/지하철역까지 거리 기반 피처를 생성합니다.

    BallTree를 사용하여 효율적으로 최근접 정류소/역을 탐색합니다.
    좌표가 실제값인지 보간값인지를 is_real_coord 플래그로 구분합니다.

    생성 피처:
        - nearest_subway_dist: 가장 가까운 지하철역 거리 (km)
        - subway_count_1km: 반경 1km 내 지하철역 수
        - subway_lines_1km: 반경 1km 내 지하철 호선 수
        - nearest_bus_dist: 가장 가까운 버스정류소 거리 (km)
        - bus_count_500m: 반경 500m 내 버스정류소 수
        - is_real_coord: 좌표가 실제값(1)인지 보간값(0)인지 플래그
    """

    @property
    def name(self) -> str:
        return "Step 6.7: 버스/지하철 거리 피처"

    @staticmethod
    def _load_transit_data(
        data_dir: str | os.PathLike,
        bus_file: str,
        subway_file: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None, pd.DataFrame | None]:
        """버스/지하철 데이터를 로드합니다.

        Returns:
            (bus_coords_rad, subway_coords_rad, subway_df)
            coords_rad는 BallTree용 라디안 좌표 [lat, lon]
        """
        from pathlib import Path

        data_path = Path(data_dir)
        bus_coords = None
        subway_coords = None
        subway_df = None

        # 버스 정류소
        bus_path = data_path / bus_file
        if bus_path.exists():
            bus_df = pd.read_csv(bus_path)
            # 좌표 컬럼명 자동 탐지 (X좌표/경도, Y좌표/위도)
            x_col = next((c for c in bus_df.columns if "X" in c.upper() or "경도" in c), None)
            y_col = next((c for c in bus_df.columns if "Y" in c.upper() or "위도" in c), None)
            if x_col and y_col:
                valid = bus_df[[x_col, y_col]].dropna()
                bus_coords = np.radians(valid[[y_col, x_col]].values)
                print(f"    버스 정류소 로드: {len(valid):,}개")
        else:
            print(f"    [WARN] 버스 데이터 없음: {bus_path}")

        # 지하철역
        subway_path = data_path / subway_file
        if subway_path.exists():
            subway_df = pd.read_csv(subway_path)
            # 좌표 컬럼명 자동 탐지
            lon_col = next((c for c in subway_df.columns if "경도" in c or "X" in c.upper()), None)
            lat_col = next((c for c in subway_df.columns if "위도" in c or "Y" in c.upper()), None)
            if lon_col and lat_col:
                valid_idx = subway_df[[lon_col, lat_col]].dropna().index
                subway_df = subway_df.loc[valid_idx].copy()
                subway_coords = np.radians(subway_df[[lat_col, lon_col]].values)
                print(f"    지하철역 로드: {len(subway_df):,}개")
        else:
            print(f"    [WARN] 지하철 데이터 없음: {subway_path}")

        return bus_coords, subway_coords, subway_df

    @staticmethod
    def _compute_features(
        df: pd.DataFrame,
        bus_coords_rad: np.ndarray | None,
        subway_coords_rad: np.ndarray | None,
        subway_df: pd.DataFrame | None,
        bus_radius_m: float,
        subway_radius_m: float,
        coord_was_missing: pd.Series | None = None,
        chunk_size: int = 50_000,
    ) -> pd.DataFrame:
        """BallTree로 거리 피처를 계산합니다 (청크 단위 처리)."""
        from sklearn.neighbors import BallTree

        df = df.copy()
        created: list[str] = []
        EARTH_RADIUS_KM = 6371.0
        n = len(df)

        if "좌표X" not in df.columns or "좌표Y" not in df.columns:
            print("    좌표 컬럼 없음 — 건너뜀")
            return df

        lon = df["좌표X"].astype("float64").values
        lat = df["좌표Y"].astype("float64").values
        query_rad = np.radians(np.column_stack([lat, lon]))

        # ── 지하철 피처 (청크 처리) ──
        if subway_coords_rad is not None and len(subway_coords_rad) > 0:
            tree = BallTree(subway_coords_rad, metric="haversine")
            radius_rad = subway_radius_m / (EARTH_RADIUS_KM * 1000)

            nearest_dist = np.zeros(n, dtype="float64")
            count_arr = np.zeros(n, dtype="int32")

            # 호선 수 계산용
            line_col = next(
                (c for c in (subway_df.columns if subway_df is not None else [])
                 if "호선" in c or "line" in c.lower()),
                None,
            )
            line_arr = np.zeros(n, dtype="int32") if line_col else None
            line_values = subway_df[line_col].values if (line_col and subway_df is not None) else None

            n_chunks = (n + chunk_size - 1) // chunk_size
            for i in range(n_chunks):
                s, e = i * chunk_size, min((i + 1) * chunk_size, n)
                chunk_q = query_rad[s:e]

                dist, _ = tree.query(chunk_q, k=1)
                nearest_dist[s:e] = dist.flatten() * EARTH_RADIUS_KM

                counts = tree.query_radius(chunk_q, r=radius_rad, count_only=True)
                count_arr[s:e] = counts

                if line_values is not None:
                    indices = tree.query_radius(chunk_q, r=radius_rad)
                    for j, idx_list in enumerate(indices):
                        if len(idx_list) > 0:
                            line_arr[s + j] = len(set(line_values[idx_list]))

                if (i + 1) % 5 == 0 or i == n_chunks - 1:
                    print(f"    지하철 피처: {e:,}/{n:,} ({100 * e / n:.0f}%)")

            df["nearest_subway_dist"] = nearest_dist
            created.append("nearest_subway_dist")
            df["subway_count_1km"] = count_arr
            created.append("subway_count_1km")
            if line_arr is not None:
                df["subway_lines_1km"] = line_arr
                created.append("subway_lines_1km")

        # ── 버스 피처 (청크 처리) ──
        if bus_coords_rad is not None and len(bus_coords_rad) > 0:
            tree = BallTree(bus_coords_rad, metric="haversine")
            radius_rad = bus_radius_m / (EARTH_RADIUS_KM * 1000)

            nearest_dist = np.zeros(n, dtype="float64")
            count_arr = np.zeros(n, dtype="int32")

            n_chunks = (n + chunk_size - 1) // chunk_size
            for i in range(n_chunks):
                s, e = i * chunk_size, min((i + 1) * chunk_size, n)
                chunk_q = query_rad[s:e]

                dist, _ = tree.query(chunk_q, k=1)
                nearest_dist[s:e] = dist.flatten() * EARTH_RADIUS_KM

                counts = tree.query_radius(chunk_q, r=radius_rad, count_only=True)
                count_arr[s:e] = counts

                if (i + 1) % 5 == 0 or i == n_chunks - 1:
                    print(f"    버스 피처: {e:,}/{n:,} ({100 * e / n:.0f}%)")

            df["nearest_bus_dist"] = nearest_dist
            created.append("nearest_bus_dist")
            df["bus_count_500m"] = count_arr
            created.append("bus_count_500m")

        # ── is_real_coord 플래그 ──
        if coord_was_missing is not None:
            df["is_real_coord"] = (~coord_was_missing).astype("int8")
            created.append("is_real_coord")

        # NaN 안전 처리
        for col in created:
            df[col] = df[col].fillna(0)

        print(f"    생성된 교통 피처 ({len(created)}개): {created}")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cfg = ctx.config

        # 좌표 보간 전 결측 여부 기록 (is_real_coord 플래그용)
        # 좌표 보간은 Step 6에서 이미 완료됨 → train_stats에서 복원
        train_coord_missing = None
        test_coord_missing = None
        if "좌표X" in ctx.X_train.columns:
            # 보간 후이므로 NaN이면 보간도 실패한 것
            # 대부분은 보간으로 채워짐 → 원본 결측 여부는 missing_count로 간접 추정
            # 더 정확하게는 coord_group_means 존재 여부로 판단
            # 여기서는 남아있는 NaN만 체크 (보간 실패 건)
            train_coord_missing = ctx.X_train["좌표X"].isna()
            test_coord_missing = ctx.X_test["좌표X"].isna()

        # 데이터 로드
        print("  교통 데이터 로드:")
        bus_coords, subway_coords, subway_df = self._load_transit_data(
            cfg.data_dir,
            cfg.bus_feature_file,
            cfg.subway_feature_file,
        )

        if bus_coords is None and subway_coords is None:
            print("  [WARN] 버스/지하철 데이터 모두 없음 — 건너뜀")
            return ctx

        # 학습 데이터
        print("  학습 데이터:")
        ctx.X_train = self._compute_features(
            ctx.X_train, bus_coords, subway_coords, subway_df,
            cfg.transit_bus_radius_m, cfg.transit_subway_radius_m,
            train_coord_missing,
        )

        # 테스트 데이터
        print("  테스트 데이터:")
        ctx.X_test = self._compute_features(
            ctx.X_test, bus_coords, subway_coords, subway_df,
            cfg.transit_bus_radius_m, cfg.transit_subway_radius_m,
            test_coord_missing,
        )

        print(f"  현재 컬럼 수: X_train={ctx.X_train.shape[1]}, X_test={ctx.X_test.shape[1]}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 7: 결측값 대체 (KNN Imputer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MissingValueImputerStep(PreprocessingStep):
    """범주형 → '미상' 대체, 수치형 → KNN Imputer로 결측을 보간합니다."""

    @property
    def name(self) -> str:
        return "Step 7: 결측값 대체 (KNN Imputer)"

    @staticmethod
    def _handle_missing(
        df: pd.DataFrame,
        categorical_cols: list[str],
        n_neighbors: int = 5,
        sample_size: int = 30_000,
        chunk_size: int = 50_000,
        fitted_medians: dict | None = None,
        numeric_cols_order: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict | None, list[str] | None]:
        """결측값 처리: 범주형→'미상', 수치형→중앙값 대체.

        트리 기반 모델(LightGBM, XGBoost, CatBoost)은 결측값을 자체 처리하지만,
        일부 파생 피처 계산 시 NaN이 전파되므로 중앙값으로 빠르게 대체합니다.
        """
        df = df.copy()

        # 1) 범주형 → '미상' (문자열 dtype만, 숫자형 범주는 median으로 처리)
        cat_filled = 0
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == object:
                n_na = df[col].isna().sum()
                if n_na > 0:
                    df[col] = df[col].fillna("미상")
                    cat_filled += n_na
        print(f"  범주형 결측 대체: {cat_filled:,}건 → '미상' (문자열 dtype만)")

        # 2) 수치형 → 중앙값(Median) 대체
        if numeric_cols_order is not None:
            num_cols = [c for c in numeric_cols_order if c in df.columns]
        else:
            num_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in categorical_cols
            ]

        # Nullable Int64 → float64 변환 (pd.NA → np.nan)
        for col in num_cols:
            if hasattr(df[col], "dtype") and str(df[col].dtype) in ("Int8", "Int16", "Int32", "Int64"):
                df[col] = df[col].astype("float64")

        total_na = df[num_cols].isna().sum().sum()
        if total_na == 0:
            print("  수치형 결측 없음 — 건너뜀")
            return df, fitted_medians, num_cols

        print(f"  수치형 결측: {total_na:,}건 → Median Imputer")

        # Fit (학습 데이터에서만)
        if fitted_medians is None:
            medians = {}
            for col in num_cols:
                med = df[col].median()
                medians[col] = med
            print(f"  Median 계산 완료: {len(medians)}개 컬럼")
        else:
            medians = fitted_medians

        # Transform
        filled_count = 0
        for col in num_cols:
            n_na = df[col].isna().sum()
            if n_na > 0 and col in medians:
                df[col] = df[col].fillna(medians[col])
                filled_count += n_na

        remaining_na = df[num_cols].isna().sum().sum()
        print(f"  Median 완료 — 대체: {filled_count:,}건, 잔여 결측: {remaining_na:,}건")

        return df, medians, num_cols

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cfg = ctx.config

        # 학습 데이터
        print("  학습 데이터:")
        ctx.X_train, medians, num_cols = self._handle_missing(
            ctx.X_train,
            ctx.categorical_cols,
        )
        ctx.train_stats["median_values"] = medians
        ctx.train_stats["median_numeric_cols"] = num_cols

        # 테스트 데이터 (학습 기준 median 재사용)
        print("\n  테스트 데이터 (학습 기준 Median):")
        ctx.X_test, _, _ = self._handle_missing(
            ctx.X_test,
            ctx.categorical_cols,
            fitted_medians=ctx.train_stats.get("median_values"),
            numeric_cols_order=ctx.train_stats.get("median_numeric_cols"),
        )
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 7.5: 세대당 주차대수 파생 피처
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ParkingPerHouseholdStep(PreprocessingStep):
    """주차대수 / k-전체세대수 = 세대당 주차대수 피처를 생성합니다."""

    @property
    def name(self) -> str:
        return "Step 7.5: 세대당 주차대수 파생"

    @staticmethod
    def _add_feature(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        parking_col = "주차대수"
        household_col = "k-전체세대수"

        if parking_col in df.columns and household_col in df.columns:
            parking = df[parking_col].fillna(0)
            household = df[household_col].fillna(0)
            df["parking_per_household"] = parking / (household + 1e-6)
            df["parking_per_household"] = (
                df["parking_per_household"]
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )
            print(f"    parking_per_household 생성 완료")
            print(
                f"    통계: mean={df['parking_per_household'].mean():.4f}, "
                f"median={df['parking_per_household'].median():.4f}, "
                f"max={df['parking_per_household'].max():.4f}"
            )
        else:
            missing = [c for c in [parking_col, household_col] if c not in df.columns]
            print(f"    건너뜀 — 컬럼 없음: {missing}")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        print("  학습 데이터:")
        ctx.X_train = self._add_feature(ctx.X_train)
        print("  테스트 데이터:")
        ctx.X_test = self._add_feature(ctx.X_test)
        print(f"  현재 컬럼 수: X_train={ctx.X_train.shape[1]}, X_test={ctx.X_test.shape[1]}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 7.7: 교호작용 / 비율 파생 피처
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InteractionFeaturesStep(PreprocessingStep):
    """수치형 피처 간 교호작용/비율/변환 피처를 생성합니다.

    트리 모델은 피처 간 곱/나눗셈을 직접 학습하기 어렵기 때문에
    도메인 지식 기반의 파생 피처를 명시적으로 생성합니다.

    생성 피처:
        - 면적x층: 전용면적 × 층 (고층 대형 프리미엄)
        - 면적_건물나이비: 전용면적 / (건물나이+1) (신축 대형의 가치)
        - log_전용면적: log1p(전용면적) (비선형 면적 효과)
        - 전용면적_sq: 전용면적² (면적 증가에 따른 가속 가격 상승)
        - 층_건물나이비: 층 / (건물나이+1) (신축 고층 프리미엄)
        - 동당세대수: k-전체세대수 / (k-전체동수+1) (단지 밀집도)
    """

    @property
    def name(self) -> str:
        return "Step 7.7: 교호작용 피처"

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        created: list[str] = []

        # 전용면적 × 층 (고층 대형 프리미엄)
        if "전용면적" in df.columns and "층" in df.columns:
            area = pd.to_numeric(df["전용면적"], errors="coerce")
            floor = pd.to_numeric(df["층"], errors="coerce")
            df["면적x층"] = area * floor
            created.append("면적x층")

        # 전용면적 / (건물나이 + 1) (신축 대형 가치)
        if "전용면적" in df.columns and "건물나이" in df.columns:
            area = pd.to_numeric(df["전용면적"], errors="coerce")
            age = pd.to_numeric(df["건물나이"], errors="coerce")
            df["면적_건물나이비"] = area / (age + 1)
            created.append("면적_건물나이비")

        # log(전용면적 + 1) (비선형 면적 효과)
        if "전용면적" in df.columns:
            area = pd.to_numeric(df["전용면적"], errors="coerce")
            df["log_전용면적"] = np.log1p(area)
            created.append("log_전용면적")

        # 전용면적² (가속 가격 상승)
        if "전용면적" in df.columns:
            area = pd.to_numeric(df["전용면적"], errors="coerce")
            df["전용면적_sq"] = area ** 2
            created.append("전용면적_sq")

        # 층 / (건물나이 + 1)
        if "층" in df.columns and "건물나이" in df.columns:
            floor = pd.to_numeric(df["층"], errors="coerce")
            age = pd.to_numeric(df["건물나이"], errors="coerce")
            df["층_건물나이비"] = floor / (age + 1)
            created.append("층_건물나이비")

        # 동당 세대수 (단지 밀집도)
        if "k-전체세대수" in df.columns and "k-전체동수" in df.columns:
            household = pd.to_numeric(df["k-전체세대수"], errors="coerce").fillna(0)
            buildings = pd.to_numeric(df["k-전체동수"], errors="coerce").fillna(0)
            df["동당세대수"] = household / (buildings + 1)
            created.append("동당세대수")

        # inf / NaN 안전 처리
        for col in created:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"  생성된 교호작용 피처 ({len(created)}개): {created}")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        print("  학습 데이터:")
        ctx.X_train = self._add_features(ctx.X_train)
        print("  테스트 데이터:")
        ctx.X_test = self._add_features(ctx.X_test)
        print(f"  현재 컬럼 수: X_train={ctx.X_train.shape[1]}, X_test={ctx.X_test.shape[1]}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 7.8: Bayesian Smoothed Target Encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TargetEncodingStep(PreprocessingStep):
    """고카디널리티 범주형 피처를 Bayesian Smoothed Target Encoding합니다.

    LightGBM의 native categorical 처리는 고유값 20,000+인 피처에서
    효율이 떨어질 수 있습니다. Target Encoding으로 가격 신호를
    직접적인 수치형 피처로 변환합니다.

    인코딩 공식 (Bayesian Smoothing):
        te_value = (count × group_mean + α × global_mean) / (count + α)

        - count: 해당 범주의 학습 데이터 건수
        - group_mean: 해당 범주의 log1p(target) 평균
        - global_mean: 전체 학습 데이터의 log1p(target) 평균
        - α (smoothing): 사전 분포 강도 (건수가 적을수록 전역 평균에 수렴)

    데이터 누수 방지:
        - 학습 데이터에서 encoding map 생성
        - 테스트 데이터에는 학습 기준 map 적용
        - 테스트에 없는 범주 → 전역 평균 대체
    """

    @property
    def name(self) -> str:
        return "Step 7.8: Target Encoding"

    @staticmethod
    def _encode(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        target_encode_cols: list[str],
        smoothing: int = 100,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Bayesian Smoothed Target Encoding을 수행합니다.

        log1p(target) 공간에서 인코딩하여 스케일 왜곡을 방지합니다.
        """
        X_train = X_train.copy()
        X_test = X_test.copy()
        encoding_maps: dict[str, dict] = {}
        created: list[str] = []

        # log1p 공간에서 인코딩 (모델이 log 타겟을 학습하므로)
        y_log = np.log1p(y_train.values)
        global_mean = float(np.mean(y_log))

        for col in target_encode_cols:
            if col not in X_train.columns:
                print(f"    {col}: 컬럼 없음 — 건너뜀")
                continue

            # 그룹별 통계 계산
            tmp = pd.DataFrame({"key": X_train[col].astype(str), "y": y_log})
            group_stats = tmp.groupby("key")["y"].agg(["mean", "count"])

            # Bayesian Smoothed Mean
            smoothed = (
                group_stats["count"] * group_stats["mean"]
                + smoothing * global_mean
            ) / (group_stats["count"] + smoothing)

            encoding_map = smoothed.to_dict()
            encoding_maps[col] = {"map": encoding_map, "global_mean": global_mean}

            # 인코딩 적용
            new_col = f"te_{col}"
            X_train[new_col] = (
                X_train[col].astype(str).map(encoding_map).fillna(global_mean)
            )
            X_test[new_col] = (
                X_test[col].astype(str).map(encoding_map).fillna(global_mean)
            )
            created.append(new_col)

            train_nunique = X_train[col].nunique()
            test_coverage = X_test[col].astype(str).isin(encoding_map).mean()
            print(
                f"    {new_col}: 학습 고유값={train_nunique:,}, "
                f"테스트 커버리지={test_coverage:.1%}"
            )

        # 빈도 인코딩 (거래 건수 = 인기 지표)
        for col in target_encode_cols:
            if col not in X_train.columns:
                continue
            freq_col = f"freq_{col}"
            freq_map = X_train[col].astype(str).value_counts().to_dict()
            X_train[freq_col] = X_train[col].astype(str).map(freq_map).fillna(0).astype("float64")
            X_test[freq_col] = X_test[col].astype(str).map(freq_map).fillna(0).astype("float64")
            created.append(freq_col)
            encoding_maps[f"freq_{col}"] = freq_map

        print(f"  총 생성 피처: {len(created)}개")
        return X_train, X_test, encoding_maps

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cfg = ctx.config
        te_cols = getattr(cfg, "target_encode_cols", [
            "아파트명", "도로명", "번지", "시군구", "구", "동",
        ])
        te_smoothing = getattr(cfg, "target_encode_smoothing", 100)

        print(f"  대상 컬럼 ({len(te_cols)}개): {te_cols}")
        print(f"  Smoothing factor: {te_smoothing}")

        ctx.X_train, ctx.X_test, te_maps = self._encode(
            ctx.X_train,
            ctx.X_test,
            ctx.y_train,
            te_cols,
            smoothing=te_smoothing,
        )
        ctx.train_stats["target_encoding_maps"] = te_maps
        print(f"  현재 컬럼 수: X_train={ctx.X_train.shape[1]}, X_test={ctx.X_test.shape[1]}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 8: 이상치 클리핑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OutlierClippingStep(PreprocessingStep):
    """IQR 기반 이상치 클리핑 + Target 퍼센타일 클리핑."""

    @property
    def name(self) -> str:
        return "Step 8: 이상치 클리핑"

    @staticmethod
    def _clip_features(
        df: pd.DataFrame, columns: list[str], factor: float,
    ) -> pd.DataFrame:
        df = df.copy()
        total_clipped = 0
        for col in columns:
            if col not in df.columns:
                print(f"    {col}: 컬럼 없음 — 건너뜀")
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            n_below = (df[col] < lower).sum()
            n_above = (df[col] > upper).sum()
            n_clipped = n_below + n_above
            df[col] = df[col].clip(lower=lower, upper=upper)
            if n_clipped > 0:
                print(
                    f"    {col}: 클리핑 {n_clipped:,}건 "
                    f"(하한={lower:.2f}, 상한={upper:.2f})"
                )
            total_clipped += n_clipped
        print(f"    총 클리핑: {total_clipped:,}건")
        return df

    @staticmethod
    def _clip_target(
        y: pd.Series, lower_pct: float, upper_pct: float,
    ) -> pd.Series:
        lower_val = y.quantile(lower_pct)
        upper_val = y.quantile(upper_pct)
        n_below = (y < lower_val).sum()
        n_above = (y > upper_val).sum()
        print(f"  Target 클리핑 범위: [{lower_val:,.0f}, {upper_val:,.0f}] (만원)")
        print(f"    하위 {lower_pct*100}% 이하: {n_below:,}건")
        print(f"    상위 {upper_pct*100}% 이상: {n_above:,}건")
        print(f"    총 클리핑: {n_below + n_above:,}건 ({(n_below + n_above) / len(y) * 100:.3f}%)")
        return y.clip(lower=lower_val, upper=upper_val)

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cfg = ctx.config

        # 수치형 피처 클리핑 (학습 데이터만)
        print("  수치형 이상치 클리핑 (학습 데이터):")
        ctx.X_train = self._clip_features(ctx.X_train, cfg.outlier_clip_cols, cfg.outlier_iqr_factor)

        # Target 클리핑
        print("\n  Target 이상치 클리핑:")
        ctx.y_train = self._clip_target(
            ctx.y_train, cfg.target_lower_percentile, cfg.target_upper_percentile,
        )
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 9: Target 로그 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TargetLogTransformStep(PreprocessingStep):
    """Target에 log1p 변환을 적용합니다."""

    @property
    def name(self) -> str:
        return "Step 9: Target 로그 변환"

    @staticmethod
    def log_transform(y: pd.Series) -> pd.Series:
        return np.log1p(y)

    @staticmethod
    def inverse_transform(y_log: np.ndarray) -> np.ndarray:
        """log1p 역변환 (예측 시 사용)."""
        return np.expm1(y_log)

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        ctx.y_train_log = self.log_transform(ctx.y_train)

        print(f"  변환 전 - 범위: [{ctx.y_train.min():,.0f}, {ctx.y_train.max():,.0f}]")
        print(f"  변환 후 - 범위: [{ctx.y_train_log.min():.4f}, {ctx.y_train_log.max():.4f}]")
        print(f"  변환 전 왜도: {ctx.y_train.skew():.4f}")
        print(f"  변환 후 왜도: {ctx.y_train_log.skew():.4f}")

        # 역변환 검증
        y_restored = self.inverse_transform(ctx.y_train_log.values)
        max_error = np.max(np.abs(ctx.y_train.values - y_restored))
        print(f"  역변환 최대 오차: {max_error:.10f}")
        return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 8.5: 저중요도 피처 제거
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LowImportanceFeatureRemovalStep(PreprocessingStep):
    """피처 중요도가 극히 낮은 컬럼을 제거하여 노이즈를 줄입니다.

    이전 학습 결과의 feature importance에서 중요도 < 50인 피처들을 제거합니다.
    config.low_importance_cols에 제거 대상 리스트가 관리됩니다.
    """

    @property
    def name(self) -> str:
        return "Step 8.5: 저중요도 피처 제거"

    @staticmethod
    def _remove(df: pd.DataFrame, cols_to_remove: list[str]) -> pd.DataFrame:
        existing = [c for c in cols_to_remove if c in df.columns]
        if existing:
            df = df.drop(columns=existing, errors="ignore")
            print(f"    제거된 컬럼 ({len(existing)}개): {existing}")
        else:
            print("    제거할 컬럼 없음")
        return df

    def execute(self, ctx: PreprocessingContext) -> PreprocessingContext:
        cols = ctx.config.low_importance_cols
        print(f"  제거 대상: {len(cols)}개")
        print("  학습 데이터:")
        ctx.X_train = self._remove(ctx.X_train, cols)
        print("  테스트 데이터:")
        ctx.X_test = self._remove(ctx.X_test, cols)
        # 범주형 목록에서도 제거
        ctx.categorical_cols = [c for c in ctx.categorical_cols if c not in cols]
        print(f"  현재 컬럼 수: X_train={ctx.X_train.shape[1]}, X_test={ctx.X_test.shape[1]}")
        return ctx
