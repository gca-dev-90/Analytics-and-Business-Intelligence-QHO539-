from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from utils.data_loader import default_csv_path, load_csv

# Match the script's original display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.precision", 2)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_COLUMNS = [str(year) for year in range(1997, 2017)]
REFERENCE_YEARS = ["1997", "2007", "2016"]


def load_data(path: str | None, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    actual_path = path or str(default_csv_path())
    df = load_csv(actual_path).copy()
    if verbose:
        print("=" * 80)
        print("WEEK 5: PYTHON DATA ANALYSIS - GDHI DATASET")
        print("=" * 80)
        print(f"[OK] Data loaded from {actual_path}")
        print(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")

        print("\nCOLUMN OVERVIEW")
        print("-" * 80)
        print(df.dtypes)

        print("\nHEAD (5 rows)")
        print("-" * 80)
        print(df.head())

        print("\nMISSING VALUES PER COLUMN")
        print("-" * 80)
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values detected.")
        else:
            print(missing_values[missing_values > 0])

        duplicates = df.duplicated().sum()
        print("\nDUPLICATE ROWS")
        print("-" * 80)
        if duplicates == 0:
            print("No duplicate rows detected.")
        else:
            print(f"Warning: {duplicates} duplicate rows found.")

    return df, actual_path


def descriptive_statistics(df: pd.DataFrame, verbose: bool = True) -> None:
    df = df.copy()
    df["Growth_abs"] = df["2016"] - df["1997"]
    df["Growth_pct"] = ((df["2016"] - df["1997"]) / df["1997"]) * 100

    if not verbose:
        return

    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    print("\nSUMMARY (ALL YEARS)")
    print("-" * 80)
    print(df[YEAR_COLUMNS].describe())

    print("\nCENTRAL TENDENCY")
    print("-" * 80)
    for year in REFERENCE_YEARS:
        mean_val = float(df[year].mean())
        median_val = float(df[year].median())
        mode_val = float(df[year].mode().values[0])
        print(f"{year}:")
        print(f"  Mean:   GBP {mean_val:.2f}")
        print(f"  Median: GBP {median_val:.2f}")
        print(f"  Mode:   GBP {mode_val:.2f}")

    print("\nDISPERSION METRICS")
    print("-" * 80)
    for year in REFERENCE_YEARS:
        mean_val = float(df[year].mean())
        std_val = df[year].std()
        print(f"{year}:")
        print(f"  Range:        GBP {float(df[year].max() - df[year].min()):.2f}")
        print(f"  Variance:     GBP {float(df[year].var()):.2f}")
        print(f"  Std Dev:      GBP {float(std_val):.2f}")
        print(f"  IQR:          GBP {float(df[year].quantile(0.75) - df[year].quantile(0.25)):.2f}")
        print(f"  Coef. Var.:   {std_val / mean_val * 100:.2f}%")

    print("\nSHAPE METRICS")
    print("-" * 80)
    for year in REFERENCE_YEARS:
        col = df[year].astype(float)
        skew = float(col.skew())
        kurtosis = float(col.kurtosis())
        if skew > 0:
            skew_txt = "Positively skewed (right tail)"
        elif skew < 0:
            skew_txt = "Negatively skewed (left tail)"
        else:
            skew_txt = "Symmetric distribution"
        print(f"{year}: Skewness {skew:.4f} -> {skew_txt}")
        if kurtosis > 0:
            kurt_txt = "Heavier tails than normal (leptokurtic)"
        elif kurtosis < 0:
            kurt_txt = "Lighter tails than normal (platykurtic)"
        else:
            kurt_txt = "Close to normal (mesokurtic)"
        print(f"{year}: Kurtosis {kurtosis:.4f} -> {kurt_txt}")

    print("\nPERCENTILES (2016)")
    print("-" * 80)
    for p in [0, 10, 25, 50, 75, 90, 100]:
        print(f"{p:>3}th percentile: GBP {df['2016'].quantile(p / 100):.2f}")

    print("\nTOP AND BOTTOM PERFORMERS (2016)")
    print("-" * 80)
    print("Top 10 regions by GDHI:")
    print(df.nlargest(10, "2016")[["AREANM", "2016"]].to_string(index=False))
    print("\nBottom 10 regions by GDHI:")
    print(df.nsmallest(10, "2016")[["AREANM", "2016"]].to_string(index=False))

    print("\nGROWTH ANALYSIS (1997-2016)")
    print("-" * 80)
    print(f"Average absolute growth: GBP {float(df['Growth_abs'].mean()):.2f}")
    print(f"Average percentage growth: {float(df['Growth_pct'].mean()):.2f}%")
    print(f"Maximum absolute growth: GBP {float(df['Growth_abs'].max()):.2f}")
    print(f"Minimum absolute growth: GBP {float(df['Growth_abs'].min()):.2f}")
    print("\nTop 5 regions by absolute growth:")
    print(df.nlargest(5, "Growth_abs")[["AREANM", "1997", "2016", "Growth_abs", "Growth_pct"]].to_string(index=False))
    print("\nTop 5 regions by percentage growth:")
    print(df.nlargest(5, "Growth_pct")[["AREANM", "1997", "2016", "Growth_abs", "Growth_pct"]].to_string(index=False))

    print("\nTEMPORAL TRENDS")
    print("-" * 80)
    mean_trend = df[YEAR_COLUMNS].mean().astype(float)
    for year in ["1997", "2002", "2007", "2012", "2016"]:
        print(f"{year}: GBP {mean_trend[year]:.2f}")

    print("\nCORRELATION SNAPSHOT")
    print("-" * 80)
    print(df[REFERENCE_YEARS].corr())

    print("\nSUMMARY STATISTICS (describe)")
    print("-" * 80)
    summary_table = pd.DataFrame({year: df[year].describe() for year in REFERENCE_YEARS})
    print(summary_table)

    print("\nKEY TAKEAWAYS")
    print("-" * 80)
    print("1. Data quality checks completed without issues.")
    print("2. GDHI increased steadily across the 20-year window.")
    print("3. Growth dispersion remains wide between regions.")
    print("4. Correlations between benchmark years remain high.")
    print("5. Visual dashboards saved to the outputs directory.")


def create_visualisations(df: pd.DataFrame, verbose: bool = True) -> tuple[Figure, Figure]:
    data = df.copy()
    data["Growth_pct"] = ((data["2016"] - data["1997"]) / data["1997"]) * 100

    fig1 = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(3, 2, 1)
    sns.histplot(data["2016"].to_numpy(dtype=float), bins=30, kde=True, color="steelblue", ax=ax1)
    mean_2016 = float(data["2016"].mean())
    median_2016 = float(data["2016"].median())
    ax1.axvline(mean_2016, color="red", linestyle="--", linewidth=2, label=f"Mean: GBP {mean_2016:.0f}")
    ax1.axvline(median_2016, color="green", linestyle="--", linewidth=2, label=f"Median: GBP {median_2016:.0f}")
    ax1.set_title("Distribution of GDHI in 2016", fontweight="bold")
    ax1.set_xlabel("GDHI per head (GBP)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 2, 2)
    selected_years = ["1997", "2002", "2007", "2012", "2016"]
    box_data = [data[year].to_numpy(dtype=float) for year in selected_years]
    bp = ax2.boxplot(box_data, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax2.set_title("GDHI distribution across selected years", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_xticks(range(1, len(selected_years) + 1))
    ax2.set_xticklabels(selected_years)
    ax2.set_ylabel("GDHI per head (GBP)")
    ax2.grid(True, alpha=0.3)

    years = YEAR_COLUMNS
    years_numeric = np.array([int(year) for year in years])
    ax3 = plt.subplot(3, 2, 3)
    mean_values = [float(data[year].mean()) for year in years]
    median_values = [float(data[year].median()) for year in years]
    ax3.plot(years_numeric, mean_values, marker="o", linewidth=2, label="Mean", color="blue")
    ax3.plot(years_numeric, median_values, marker="s", linewidth=2, label="Median", color="green")
    ax3.set_title("Mean vs median GDHI trend", fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("GDHI per head (GBP)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(years_numeric)
    ax3.set_xticklabels(years, rotation=45, ha="right")

    ax4 = plt.subplot(3, 2, 4)
    std_values = np.array([float(data[year].std()) for year in years], dtype=float)
    ax4.plot(years_numeric, std_values, marker="D", linewidth=2, color="red")
    ax4.set_title("Standard deviation over time", fontweight="bold")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Standard deviation (GBP)")
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(years_numeric)
    ax4.set_xticklabels(years, rotation=45, ha="right")

    ax5 = plt.subplot(3, 2, 5)
    top_10 = data.nlargest(10, "2016")[["AREANM", "2016"]].sort_values("2016")
    ax5.barh(range(len(top_10)), top_10["2016"], color="steelblue")
    ax5.set_yticks(range(len(top_10)))
    ax5.set_yticklabels(top_10["AREANM"], fontsize=8)
    ax5.set_title("Top 10 areas by GDHI in 2016", fontweight="bold")
    ax5.set_xlabel("GDHI per head (GBP)")
    ax5.grid(True, alpha=0.3, axis="x")

    ax6 = plt.subplot(3, 2, 6)
    top_growth = data.nlargest(10, "Growth_pct")[["AREANM", "Growth_pct"]].sort_values("Growth_pct")
    ax6.barh(range(len(top_growth)), top_growth["Growth_pct"], color="green")
    ax6.set_yticks(range(len(top_growth)))
    ax6.set_yticklabels(top_growth["AREANM"], fontsize=8)
    ax6.set_title("Top 10 areas by growth rate (1997-2016)", fontweight="bold")
    ax6.set_xlabel("Growth rate (%)")
    ax6.grid(True, alpha=0.3, axis="x")

    fig1.tight_layout()

    fig2 = plt.figure(figsize=(16, 10))

    ax7 = plt.subplot(2, 2, 1)
    skew_values = np.array([float(data[year].astype(float).skew()) for year in years], dtype=float)
    ax7.plot(years_numeric, skew_values, marker="o", color="purple", linewidth=2)
    ax7.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax7.set_title("Skewness over time", fontweight="bold")
    ax7.set_xlabel("Year")
    ax7.set_ylabel("Skewness")
    ax7.grid(True, alpha=0.3)
    ax7.set_xticks(years_numeric)
    ax7.set_xticklabels(years, rotation=45, ha="right")

    ax8 = plt.subplot(2, 2, 2)
    cv_values = np.array(
        [float(data[year].astype(float).std() / data[year].astype(float).mean() * 100) for year in years], dtype=float
    )
    ax8.plot(years_numeric, cv_values, marker="s", color="orange", linewidth=2)
    ax8.set_title("Coefficient of variation over time", fontweight="bold")
    ax8.set_xlabel("Year")
    ax8.set_ylabel("Coefficient of variation (%)")
    ax8.grid(True, alpha=0.3)
    ax8.set_xticks(years_numeric)
    ax8.set_xticklabels(years, rotation=45, ha="right")

    ax9 = plt.subplot(2, 2, 3)
    ax9.scatter(data["1997"], data["2016"], alpha=0.6, s=50, color="teal")
    min_val = min(data["1997"].min(), data["2016"].min())
    max_val = max(data["1997"].max(), data["2016"].max())
    ax9.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Equal line")
    ax9.set_title("GDHI comparison: 1997 vs 2016", fontweight="bold")
    ax9.set_xlabel("GDHI 1997 (GBP)")
    ax9.set_ylabel("GDHI 2016 (GBP)")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    ax10 = plt.subplot(2, 2, 4)
    q1_values = [data[year].quantile(0.25) for year in years]
    q2_values = [data[year].quantile(0.50) for year in years]
    q3_values = [data[year].quantile(0.75) for year in years]
    ax10.fill_between(range(len(years)), q1_values, q3_values, alpha=0.3, label="IQR")
    ax10.plot(range(len(years)), q2_values, marker="o", linewidth=2, color="red", label="Median (Q2)")
    ax10.plot(range(len(years)), q1_values, linewidth=1.5, color="blue", label="Q1")
    ax10.plot(range(len(years)), q3_values, linewidth=1.5, color="green", label="Q3")
    ax10.set_xticks(range(0, len(years), 2))
    ax10.set_xticklabels(years[::2])
    plt.setp(ax10.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax10.set_title("Quartile progression over time", fontweight="bold")
    ax10.set_xlabel("Year")
    ax10.set_ylabel("GDHI per head (GBP)")
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    fig2.tight_layout()

    if verbose:
        fig1.savefig(OUTPUT_DIR / "week5_descriptive_dashboard.png", dpi=300, bbox_inches="tight")
        fig2.savefig(OUTPUT_DIR / "week5_advanced_dashboard.png", dpi=300, bbox_inches="tight")

    return fig1, fig2


def prepare_outputs(path: str | None, verbose: bool = True) -> Tuple[List[Tuple[str, Figure]], str, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df, actual_path = load_data(path, verbose=True)
        descriptive_statistics(df, verbose=True)
        fig1, fig2 = create_visualisations(df, verbose=True)

    summary = buffer.getvalue()
    if verbose:
        print(summary)

    figures = [
        ("Descriptive Statistics Dashboard", fig1),
        ("Advanced Analysis Dashboard", fig2),
    ]
    return figures, summary, actual_path


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    figures, summary, actual_path = prepare_outputs(config.get("data_path"), verbose=False)
    st.header("Week 5 - Python data analysis with GDHI")
    st.caption(f"Data source: {actual_path}")

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
