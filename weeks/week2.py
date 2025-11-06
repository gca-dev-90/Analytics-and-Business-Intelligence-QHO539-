"""Week 2 - Fundamental statistics analysis for the GDHI dataset."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats
from numpy.typing import NDArray

from utils.data_loader import default_csv_path, load_csv

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


Array1D = NDArray[np.float64]


class StatsContext(TypedDict):
    gdhi_2016: Array1D
    mean: float
    median: float
    mode: float


def load_data(path: str | None, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    actual_path = path or str(default_csv_path())
    df = load_csv(actual_path).copy()

    if verbose:
        print("=" * 81)
        print("WEEK 2: FUNDAMENTAL STATISTICS ANALYSIS")
        print("Dataset: Gross Disposable Household Income (GDHI) per head")
        print("=" * 81)

        print("\n" + "=" * 80)
        print("1. DATA TYPES (Learning Outcome 3)")
        print("=" * 80)
        print(f"\nDataset dimensions: {df.shape[0]} regions × {df.shape[1]} columns")
        print("\nVariable types:")
        print("  - Categorical/Text: AREANM (Area Name), AREACD (Area Code)")
        print("  - Numerical/Integer: Years 1997-2016 (GDHI values in £)")
        print("\nData types in DataFrame:")
        print(df.dtypes.value_counts())

    return df, actual_path


def compute_central_tendency(df: pd.DataFrame, verbose: bool = True) -> StatsContext:
    gdhi_2016: Array1D = df["2016"].to_numpy(dtype=float, copy=True)
    mean_2016 = float(np.mean(gdhi_2016))
    median_2016 = float(np.median(gdhi_2016))
    mode_2016 = float(stats.mode(gdhi_2016, keepdims=True).mode[0])

    if verbose:
        print("\n" + "=" * 80)
        print("2. MEASURES OF CENTRAL TENDENCY (Learning Outcome 2)")
        print("=" * 80)
        print("\nAnalyzing GDHI per head for year 2016:")
        print(f"\n  Mean (Average):   £{mean_2016:,.2f}")
        print(f"  Median (Middle):  £{median_2016:,.2f}")
        print(f"  Mode (Most freq): £{mode_2016:,.2f}")
        print("\n  Interpretation:")
        skew_dir = "positive (right)" if median_2016 < mean_2016 else "negative (left)"
        print(f"  - The average GDHI per head across all UK regions in 2016 was £{mean_2016:,.2f}")
        print(f"  - Half of the regions had GDHI below £{median_2016:,.2f} (median)")
        print(f"  - The median is {'lower' if median_2016 < mean_2016 else 'higher'} than the mean,")
        print(f"    suggesting {skew_dir} skewness")

    return {
        "gdhi_2016": gdhi_2016,
        "mean": mean_2016,
        "median": median_2016,
        "mode": mode_2016,
    }


def compute_dispersion(stats_ctx: StatsContext, verbose: bool = True) -> dict[str, float]:
    gdhi_2016 = stats_ctx["gdhi_2016"]
    mean_val = stats_ctx["mean"]

    range_2016 = float(np.max(gdhi_2016) - np.min(gdhi_2016))
    variance_2016 = float(np.var(gdhi_2016, ddof=1))
    std_2016 = float(np.std(gdhi_2016, ddof=1))
    cv_2016 = (std_2016 / mean_val) * 100 if mean_val else float("nan")

    if verbose:
        print("\n" + "=" * 80)
        print("3. MEASURES OF DISPERSION (Learning Outcome 2)")
        print("=" * 80)
        print(f"\n  Range:              £{range_2016:,.2f}")
        print(f"    (Max - Min):      £{np.max(gdhi_2016):,.2f} - £{np.min(gdhi_2016):,.2f}")
        print(f"\n  Variance:           £{variance_2016:,.2f}²")
        print(f"  Standard Deviation: £{std_2016:,.2f}")
        print(f"  Coefficient of Var: {cv_2016:.2f}%")
        lo = mean_val - std_2016
        hi = mean_val + std_2016
        print("\n  Interpretation:")
        print(f"  - There's a £{range_2016:,.2f} difference between richest and poorest regions")
        print(f"  - Standard deviation of £{std_2016:,.2f} means typical variation from mean")
        print(f"  - About 68% of regions fall within £{lo:,.2f} to £{hi:,.2f}")

    return {
        "range": range_2016,
        "variance": variance_2016,
        "std_dev": std_2016,
        "cv": cv_2016,
    }


def compute_percentiles(stats_ctx: StatsContext, verbose: bool = True) -> dict[str, float]:
    gdhi_2016 = stats_ctx["gdhi_2016"]
    q1 = float(np.percentile(gdhi_2016, 25))
    q2 = float(np.percentile(gdhi_2016, 50))
    q3 = float(np.percentile(gdhi_2016, 75))
    iqr = q3 - q1
    p90 = float(np.percentile(gdhi_2016, 90))
    p10 = float(np.percentile(gdhi_2016, 10))

    if verbose:
        print("\n" + "=" * 80)
        print("4. PERCENTILES AND QUARTILES")
        print("=" * 80)
        print(f"\n  Q1 (25th percentile): £{q1:,.2f}")
        print(f"  Q2 (50th percentile): £{q2:,.2f} (Median)")
        print(f"  Q3 (75th percentile): £{q3:,.2f}")
        print(f"  IQR (Q3 - Q1):        £{iqr:,.2f}")
        print(f"\n  10th percentile:      £{p10:,.2f}")
        print(f"  90th percentile:      £{p90:,.2f}")
        print("\n  Interpretation:")
        print(f"  - 25% of regions have GDHI below £{q1:,.2f}")
        print(f"  - 50% of regions are between £{q1:,.2f} and £{q3:,.2f} (IQR)")
        print(f"  - 25% of regions have GDHI above £{q3:,.2f}")

    return {"q1": q1, "q2": q2, "q3": q3, "iqr": iqr, "p90": p90, "p10": p10}


def regional_rankings(df: pd.DataFrame, verbose: bool = True) -> None:
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("5. TOP AND BOTTOM PERFORMING REGIONS (2016)")
    print("=" * 80)
    top_5 = df.nlargest(5, "2016")[["AREANM", "2016"]]
    print("\n  TOP 5 HIGHEST GDHI REGIONS:")
    for _, row in top_5.iterrows():
        print(f"    {row['AREANM']:40s} £{row['2016']:,}")

    bottom_5 = df.nsmallest(5, "2016")[["AREANM", "2016"]]
    print("\n  BOTTOM 5 LOWEST GDHI REGIONS:")
    for _, row in bottom_5.iterrows():
        print(f"    {row['AREANM']:40s} £{row['2016']:,}")


def temporal_analysis(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["growth_1997_2016"] = df["2016"] - df["1997"]
    df["growth_percent"] = ((df["2016"] - df["1997"]) / df["1997"]) * 100

    if verbose:
        print("\n" + "=" * 80)
        print("6. TEMPORAL ANALYSIS: 1997 vs 2016")
        print("=" * 80)
        mean_1997 = df["1997"].mean()
        mean_2016 = df["2016"].mean()
        growth_abs = mean_2016 - mean_1997
        growth_pct = ((mean_2016 - mean_1997) / mean_1997) * 100
        print(f"\n  Mean GDHI in 1997: £{mean_1997:,.2f}")
        print(f"  Mean GDHI in 2016: £{mean_2016:,.2f}")
        print(f"  Absolute growth:   £{growth_abs:,.2f}")
        print(f"  Percentage growth: {growth_pct:.2f}%")

        fastest = df.nlargest(5, "growth_percent")[["AREANM", "growth_percent", "1997", "2016"]]
        print("\n  TOP 5 FASTEST GROWING REGIONS:")
        for _, row in fastest.iterrows():
            print(f"    {row['AREANM']:40s} {row['growth_percent']:6.2f}% ("
                  f"£{row['1997']:,} → £{row['2016']:,})")

    return df


def statistical_summary(stats_ctx: StatsContext, verbose: bool = True) -> None:
    if not verbose:
        return

    gdhi_2016 = stats_ctx["gdhi_2016"]
    print("\n" + "=" * 80)
    print("7. COMPREHENSIVE STATISTICAL SUMMARY")
    print("=" * 80)
    print("\n", pd.Series(gdhi_2016).describe())
    skewness = stats.skew(gdhi_2016)
    kurtosis = stats.kurtosis(gdhi_2016)
    print(f"\n  Skewness:  {skewness:.4f}")
    if skewness > 0:
        print("    → Positive skew: Distribution has a long right tail")
        print("    → More regions with lower GDHI; few high-income regions")
    elif skewness < 0:
        print("    → Negative skew: Distribution has a long left tail")
    else:
        print("    → Symmetric distribution")

    print(f"\n  Kurtosis:  {kurtosis:.4f}")
    if kurtosis > 0:
        print("    → Leptokurtic: Distribution has heavy tails and sharp peak")
    elif kurtosis < 0:
        print("    → Platykurtic: Distribution has light tails and flat peak")
    else:
        print("    → Mesokurtic: Normal-like distribution")


def create_dashboard_figure(df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Week 2: Descriptive Statistics - GDHI Analysis 2016", fontsize=16, fontweight="bold")

    gdhi_2016 = df["2016"]
    mean_2016 = float(np.mean(gdhi_2016))
    median_2016 = float(np.median(gdhi_2016))
    axes[0, 0].hist(gdhi_2016, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(mean_2016, color="red", linestyle="--", linewidth=2, label=f"Mean: £{mean_2016:,.0f}")
    axes[0, 0].axvline(median_2016, color="green", linestyle="--", linewidth=2, label=f"Median: £{median_2016:,.0f}")
    axes[0, 0].set_title("Distribution of GDHI (2016)", fontweight="bold")
    axes[0, 0].legend()

    axes[0, 1].boxplot(gdhi_2016, vert=True, patch_artist=True, boxprops=dict(facecolor="lightcoral"))
    axes[0, 1].set_title("Box Plot - Measures of Dispersion", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    years = [str(year) for year in range(1997, 2017)]
    mean_values = [df[year].mean() for year in years]
    axes[1, 0].plot(years, mean_values, marker="o", linewidth=2, color="darkblue")
    axes[1, 0].set_title("Trend of Average GDHI (1997-2016)", fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    std_values = [df[year].std() for year in years]
    axes[1, 1].plot(years, std_values, marker="s", linewidth=2, color="darkgreen")
    axes[1, 1].set_title("Regional Inequality Over Time", fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def print_key_insights(verbose: bool = True) -> None:
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("8. KEY INSIGHTS - WEEK 2 APPLICATION")
    print("=" * 80)
    print(
        """
  Data Types Identified:
    ✓ Categorical: Region names and codes
    ✓ Numerical (Continuous): GDHI values in pounds
    ✓ Time series: 20 years of annual data

  Central Tendency:
    ✓ Mean provides overall average income level
    ✓ Median shows the 'typical' region (middle value)
    ✓ Mean > Median indicates positive skew (income inequality)

  Dispersion Analysis:
    ✓ High standard deviation indicates significant regional inequality
    ✓ Large range shows extreme differences between richest/poorest regions
    ✓ IQR captures spread of middle 50% of regions

  Practical Applications:
    ✓ Policy makers can identify regions needing support
    ✓ Economic trends visible through time series analysis
    ✓ Inequality measured through dispersion metrics
"""
    )
    print("=" * 80)
    print("Analysis complete - Week 2 concepts successfully applied!")
    print("=" * 80)


def create_visualisations(df: pd.DataFrame, verbose: bool = True) -> Figure:
    fig = create_dashboard_figure(df)
    if verbose:
        fig.savefig(OUTPUT_DIR / "week2_dashboard.png", dpi=300, bbox_inches="tight")
    return fig


def prepare_outputs(path: str | None, verbose: bool = True):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df, actual_path = load_data(path, verbose=True)
        stats_ctx = compute_central_tendency(df, verbose=True)
        compute_dispersion(stats_ctx, verbose=True)
        compute_percentiles(stats_ctx, verbose=True)
        regional_rankings(df, verbose=True)
        df = temporal_analysis(df, verbose=True)
        statistical_summary(stats_ctx, verbose=True)
        fig = create_visualisations(df, verbose=True)
        print_key_insights(verbose=True)
    summary = buffer.getvalue()
    if verbose:
        print(summary)
    figures = [("Fundamental Statistics Dashboard", fig)]
    return figures, summary, actual_path


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    figures, summary, actual_path = prepare_outputs(config.get("data_path"), verbose=False)
    st.header("Week 2 - Fundamental statistics analysis")
    st.caption(f"Data source: {actual_path}")
    label, fig = figures[0]
    st.pyplot(fig)
    st.markdown("**Summary**")
    st.code(summary, language="text")


def build_widget(config: dict):
    figures, summary, _ = prepare_outputs(config.get("data_path"), verbose=False)
    return {"figures": figures, "text": [("Summary", summary)]}


def main() -> None:
    prepare_outputs(None, verbose=True)
    plt.show(block=False)
    print("\nAnalysis complete! Figures saved to outputs/.")


if __name__ == "__main__":
    main()
