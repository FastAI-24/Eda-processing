"""
전처리 시각화 유틸리티

노트북의 EDA/전처리 시각화 코드를 독립 모듈로 분리합니다.
관심사의 분리(Separation of Concerns) 원칙에 따라,
전처리 로직과 시각화 로직을 분리합니다.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── 전역 스타일 설정 ──
VIZ_COLORS = {
    "primary": "#3498db",
    "secondary": "#e67e22",
    "success": "#27ae60",
    "danger": "#e74c3c",
    "gray": "#95a5a6",
    "palette": ["#3498db", "#e67e22", "#27ae60", "#9b59b6", "#1abc9c"],
}


def setup_plot_style() -> None:
    """matplotlib 전역 스타일을 설정합니다."""
    warnings.filterwarnings("ignore")
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 100)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.edgecolor": "#2c3e50",
        "axes.linewidth": 0.8,
        "xtick.color": "#34495e",
        "ytick.color": "#34495e",
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDA 시각화 함수들
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_target_distribution(y: pd.Series, title_prefix: str = "") -> None:
    """Target 분포 (원본 vs log1p)를 시각화합니다."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(y, bins=100, color=VIZ_COLORS["primary"], edgecolor="white", alpha=0.85)
    axes[0].set_title("target 분포 (원본)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("거래가격 (만원)")
    axes[0].set_ylabel("빈도")

    axes[1].hist(np.log1p(y), bins=100, color=VIZ_COLORS["secondary"], edgecolor="white", alpha=0.85)
    axes[1].set_title("target 분포 (log1p 변환)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log1p(거래가격)")
    axes[1].set_ylabel("빈도")

    plt.suptitle(f"{title_prefix}Target 분포 확인", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print(f"target 왜도(skewness): {y.skew():.4f}")
    print(f"log1p(target) 왜도: {np.log1p(y).skew():.4f}")


def plot_missing_ratio(df: pd.DataFrame) -> None:
    """결측 비율을 시각화합니다."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        "결측 수": missing,
        "결측 비율(%)": missing_pct,
    }).sort_values("결측 비율(%)", ascending=False)

    print("결측 비율이 높은 컬럼 (상위 20개):")
    print(missing_info[missing_info["결측 비율(%)"] > 0].head(20))

    missing_cols = missing_info[missing_info["결측 비율(%)"] > 0]
    if len(missing_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, len(missing_cols) * 0.35)))
        ax.barh(
            missing_cols.index, missing_cols["결측 비율(%)"],
            color=VIZ_COLORS["secondary"], edgecolor="white", alpha=0.85,
        )
        ax.set_xlabel("결측 비율 (%)")
        ax.set_title("컬럼별 결측 비율", fontsize=13, fontweight="bold")
        ax.axvline(x=80, color=VIZ_COLORS["danger"], linestyle="--", linewidth=2, label="제거 임계값 (80%)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()


def plot_numeric_features(train_df: pd.DataFrame) -> None:
    """주요 수치형 피처 분포 및 Target 상관관계를 시각화합니다."""
    num_features = {
        "전용면적(㎡)": "전용면적 (㎡)",
        "층": "층",
        "건축년도": "건축년도",
        "주차대수": "주차대수",
        "k-전체세대수": "전체세대수",
    }
    available = {k: v for k, v in num_features.items() if k in train_df.columns}
    if not available:
        return

    fig, axes = plt.subplots(2, len(available), figsize=(5 * len(available), 9))

    for i, (col, label) in enumerate(available.items()):
        axes[0, i].hist(
            train_df[col].dropna(), bins=60,
            color=VIZ_COLORS["primary"], edgecolor="white", alpha=0.85,
        )
        axes[0, i].set_title(f"{label} 분포", fontsize=12, fontweight="bold")
        axes[0, i].set_xlabel(label)

        if "target" in train_df.columns:
            sample = train_df[[col, "target"]].dropna().sample(min(10000, len(train_df)), random_state=42)
            axes[1, i].scatter(sample[col], sample["target"], alpha=0.12, s=5, color=VIZ_COLORS["secondary"])
            axes[1, i].set_title(f"{label} vs target", fontsize=12)
            axes[1, i].set_xlabel(label)
            axes[1, i].set_ylabel("target (만원)")

    plt.suptitle("주요 수치형 피처 분포 및 Target 관계", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(train_df: pd.DataFrame) -> None:
    """수치형 피처 간 상관관계 히트맵을 시각화합니다."""
    key_num_cols = [
        "전용면적(㎡)", "층", "건축년도", "계약년월", "본번", "부번",
        "k-전체동수", "k-전체세대수", "k-연면적", "k-주거전용면적",
        "건축면적", "주차대수", "좌표X", "좌표Y", "target",
    ]
    available = [c for c in key_num_cols if c in train_df.columns]
    if len(available) < 2:
        return

    corr_matrix = train_df[available].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}, linecolor="white",
    )
    ax.set_title("주요 수치형 피처 상관관계 히트맵", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_coordinate_interpolation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    coord_missing_before: int,
    coord_missing_after: int,
) -> None:
    """좌표 보간 전후 비교 및 지리적 분포를 시각화합니다."""
    plot_df = X_train[["좌표X", "좌표Y"]].copy()
    plot_df["target"] = y_train.values
    plot_df = plot_df.dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    valid_before = len(X_train) - coord_missing_before
    valid_after = len(X_train) - coord_missing_after
    bars = axes[0].bar(
        ["보간 전\n(유효)", "보간 후\n(유효)"],
        [valid_before, valid_after],
        color=[VIZ_COLORS["gray"], VIZ_COLORS["success"]],
        edgecolor="white", alpha=0.9,
    )
    axes[0].set_title("좌표 보간 전후 비교", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("행 수")
    for b in bars:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 5000, f"{int(h):,}", ha="center", fontsize=11)
    axes[0].set_ylim(0, max(valid_before, valid_after) * 1.15)

    sample = plot_df.sample(min(50000, len(plot_df)), random_state=42)
    h = axes[1].hexbin(sample["좌표X"], sample["좌표Y"], gridsize=80, cmap="Blues", mincnt=1)
    axes[1].set_title("거래 밀집도 (보간 후)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("경도 (좌표X)")
    axes[1].set_ylabel("위도 (좌표Y)")
    plt.colorbar(h, ax=axes[1], label="거래 건수")

    sc = axes[2].scatter(
        sample["좌표X"], sample["좌표Y"],
        c=np.log1p(sample["target"]), cmap="RdYlGn_r", s=2, alpha=0.6,
    )
    axes[2].set_title("지역별 가격 분포 (log1p 스케일)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("경도 (좌표X)")
    axes[2].set_ylabel("위도 (좌표Y)")
    plt.colorbar(sc, ax=axes[2], label="log1p(target)")

    plt.suptitle("좌표 결측 보간 결과", fontsize=15, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print(f"보간 전 결측: {coord_missing_before:,}건 → 보간 후 유효: {len(plot_df):,}건")


def plot_target_clipping(
    y_before: pd.Series,
    y_after: pd.Series,
    lower_val: float,
    upper_val: float,
) -> None:
    """Target 이상치 클리핑 전후 비교를 시각화합니다."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(y_before, bins=100, color=VIZ_COLORS["primary"], edgecolor="white", alpha=0.85)
    axes[0].set_title("target 분포 (클리핑 전)", fontsize=13, fontweight="bold")
    axes[0].axvline(lower_val, color=VIZ_COLORS["danger"], linestyle="--", linewidth=2, label=f"하한: {lower_val:,.0f}")
    axes[0].axvline(upper_val, color=VIZ_COLORS["danger"], linestyle="--", linewidth=2, label=f"상한: {upper_val:,.0f}")
    axes[0].legend()

    axes[1].hist(y_after, bins=100, color=VIZ_COLORS["success"], edgecolor="white", alpha=0.85)
    axes[1].set_title("target 분포 (클리핑 후)", fontsize=13, fontweight="bold")

    plt.suptitle("Target 이상치 클리핑 (퍼센타일 기반)", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_train_test_comparison(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
) -> None:
    """Train/Test 주요 피처 분포 비교를 시각화합니다."""
    compare_cols = ["전용면적(㎡)", "층", "건축년도", "계약년월"]
    available = [c for c in compare_cols if c in train_df.columns and c in test_df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        ax.hist(train_df[col].dropna(), bins=50, alpha=0.6, label="Train", density=True, color=VIZ_COLORS["primary"], edgecolor="white")
        ax.hist(test_df[col].dropna(), bins=50, alpha=0.6, label="Test", density=True, color=VIZ_COLORS["danger"], edgecolor="white")
        ax.set_title(f"{col}", fontsize=12, fontweight="bold")
        ax.legend()
        ax.set_ylabel("밀도")

    plt.suptitle("Train vs Test 주요 피처 분포 비교", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.show()
