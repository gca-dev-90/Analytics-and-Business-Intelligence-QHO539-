"""Week 4 - Volatility and resilience analysis for the GDHI dataset."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from utils.data_loader import default_csv_path, load_csv

YEAR_COLUMNS = [str(year) for year in range(1997, 2017)]
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: str | None, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Load the CSV and compute volatility/resilience indicators."""

    actual_path = path or str(default_csv_path())
    df = load_csv(actual_path).copy()
    df["Growth_pct"] = ((df["2016"] - df["1997"]) / df["1997"]) * 100
    df["Growth_gbp"] = df["2016"] - df["1997"]

    span = len(YEAR_COLUMNS) - 1
    ratio = (df["2016"] / df["1997"]).replace([np.inf, -np.inf], np.nan)
    df["CAGR_pct"] = (ratio ** (1 / span) - 1) * 100
    df["Volatility"] = df[YEAR_COLUMNS].std(axis=1)

    x = np.arange(len(YEAR_COLUMNS))
    slopes = []
    for _, row in df[YEAR_COLUMNS].iterrows():
        slope, _ = np.polyfit(x, row.values, 1) # type: ignore
        slopes.append(slope)
    df["Trend_slope"] = slopes
    df["Resilience"] = df["Trend_slope"] / df["Volatility"].replace({0: np.nan})

    if verbose:
        print("=" * 80)
        print("WEEK 4: VOLATILITY AND RESILIENCE ANALYSIS")
        print("=" * 80)
        print(f"[OK] Data loaded from {actual_path}")
        print(f"Rows: {len(df)}  Columns: {len(df.columns)}")

    return df, actual_path


def create_visualisations(df: pd.DataFrame) -> Figure:
    """Compose a volatility and resilience dashboard."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_vol, ax_scatter, ax_corr, ax_trend = axes.flatten()

    ax_vol.hist(df["Volatility"], bins=30, color="#4472C4", alpha=0.8, edgecolor="black")
    ax_vol.set_title("Distribution of volatility (1997-2016)", fontweight="bold")
    ax_vol.set_xlabel("GDHI volatility (GBP)")
    ax_vol.set_ylabel("Number of areas")
    ax_vol.grid(True, alpha=0.3)

    ax_scatter.scatter(
        df["Volatility"],
        df["Growth_pct"],
        c=df["Resilience"],
        cmap="viridis",
        alpha=0.8,
    )
    ax_scatter.set_title("Growth vs volatility (1997-2016)", fontweight="bold")
    ax_scatter.set_xlabel("Volatility (standard deviation of GDHI)")
    ax_scatter.set_ylabel("Growth 1997-2016 (%)")
    scatter_cb = fig.colorbar(ax_scatter.collections[0], ax=ax_scatter, label="Resilience score")
    scatter_cb.ax.set_ylabel("Trend slope / volatility")

    corr = df[YEAR_COLUMNS].corr()
    im = ax_corr.imshow(corr.values, cmap="coolwarm", vmin=0.6, vmax=1.0)
    tick_positions_x = list(range(0, len(YEAR_COLUMNS), 2))
    tick_positions_y = list(range(0, len(YEAR_COLUMNS), 4))
    ax_corr.set_xticks(tick_positions_x)
    ax_corr.set_xticklabels([YEAR_COLUMNS[i] for i in tick_positions_x], rotation=45)
    ax_corr.set_yticks(tick_positions_y)
    ax_corr.set_yticklabels([YEAR_COLUMNS[i] for i in tick_positions_y])
    ax_corr.set_title("Correlation of annual GDHI series", fontweight="bold")
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    top_resilient = df.nlargest(5, "Resilience")
    ax_trend.set_title("Resilient areas vs national average", fontweight="bold")
    ax_trend.plot(
        YEAR_COLUMNS,
        df[YEAR_COLUMNS].mean(),
        color="#4e79a7",
        linewidth=3,
        label="National average",
    )
    for _, row in top_resilient.iterrows():
        ax_trend.plot(YEAR_COLUMNS, row[YEAR_COLUMNS], alpha=0.7, linewidth=1.5, label=row["AREANM"])
    ax_trend.set_ylabel("GDHI per head (GBP)")
    ax_trend.tick_params(axis="x", rotation=45)
    ax_trend.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig


def _print_summary(df: pd.DataFrame) -> None:
    print("\nOVERVIEW")
    print("-" * 72)
    print(f"Median volatility: {df['Volatility'].median():.2f} GBP")
    print(f"Median CAGR: {df['CAGR_pct'].median():.2f}%")
    print(f"Median resilience: {df['Resilience'].median():.3f}")

    print("\nTOP 5 RESILIENT REGIONS")
    print("-" * 72)
    top_resilient = (
        df[["AREANM", "Resilience", "Growth_pct", "Volatility"]]
        .dropna(subset=["Resilience"])
        .nlargest(5, "Resilience")
    )
    for _, row in top_resilient.iterrows():
        print(
            f"- {row['AREANM']}: resilience {row['Resilience']:.3f}, "
            f"volatility {row['Volatility']:.2f} GBP, growth {row['Growth_pct']:.1f}%"
        )

    print("\nLEAST RESILIENT REGIONS")
    print("-" * 72)
    least_resilient = (
        df[["AREANM", "Resilience", "Growth_pct", "Volatility"]]
        .dropna(subset=["Resilience"])
        .nsmallest(5, "Resilience")
    )
    for _, row in least_resilient.iterrows():
        print(
            f"- {row['AREANM']}: resilience {row['Resilience']:.3f}, "
            f"volatility {row['Volatility']:.2f} GBP, growth {row['Growth_pct']:.1f}%"
        )

    print("\nSPREAD OF GROWTH RATES")
    print("-" * 72)
    print(f"Average growth: {df['Growth_pct'].mean():.2f}%")
    print(f"90th percentile growth: {df['Growth_pct'].quantile(0.9):.2f}%")
    print(f"10th percentile growth: {df['Growth_pct'].quantile(0.1):.2f}%")


def prepare_outputs(path: str | None, verbose: bool = True) -> Tuple[List[Tuple[str, Figure]], str, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df, actual_path = load_data(path, verbose=True)
        _print_summary(df)
        fig = create_visualisations(df)

    summary = buffer.getvalue()
    if verbose:
        print(summary)
        fig.savefig(OUTPUT_DIR / "week4_volatility_dashboard.png", dpi=300, bbox_inches="tight")

    figures = [("Volatility & Resilience Dashboard", fig)]
    return figures, summary, actual_path


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    figures, summary, actual_path = prepare_outputs(config.get("data_path"), verbose=False)
    st.header("Week 4 - Volatility & Resilience analysis")
    st.caption(f"Data source: {actual_path}")

    if len(figures) == 1:
        label, figure = figures[0]
        st.pyplot(figure)
    else:
        labels = [title for title, _ in figures]
        choice = st.selectbox("Select visualisation", labels)
        fig_map = dict(figures)
        st.pyplot(fig_map[choice])

    st.markdown("**Summary**")
    st.code(summary, language="text")


def build_widget(config: dict):
    figures, summary, _ = prepare_outputs(config.get("data_path"), verbose=False)
    return {"figures": figures, "text": [("Summary", summary)]}


def main() -> None:
    prepare_outputs(None, verbose=True)


if __name__ == "__main__":
    main()
