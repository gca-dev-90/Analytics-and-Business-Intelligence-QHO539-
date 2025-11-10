from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from utils.data_loader import default_csv_path, load_csv
from utils.qt_mpl import MplWidget  # type: ignore

# Match the original script's display preferences when run from the CLI.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def _ensure_output_dir(output_dir: Path | None) -> Path:
    target = output_dir or Path("outputs")
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_data(path: str | None, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Load the GDHI dataset (1997-2016)."""

    actual_path = path or str(default_csv_path())
    df = load_csv(actual_path).copy()
    if verbose:
        print("Loading data...")
        print(f"Loaded {len(df)} regions with data from 1997-2016")
    return df, actual_path


def create_tableau_ready_data(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert the wide format to long format (Tableau friendly)."""

    target_dir = _ensure_output_dir(output_dir)
    year_cols = [str(year) for year in range(1997, 2017)]

    df_tableau = df.melt(
        id_vars=["AREANM", "AREACD"],
        value_vars=year_cols,
        var_name="Year",
        value_name="GDHI_Per_Head",
    )
    df_tableau["Year"] = df_tableau["Year"].astype(int)
    df_tableau = df_tableau.sort_values(["AREANM", "Year"])

    # Always save to CSV
    output_file = target_dir / "gdhi_tableau_format.csv"
    df_tableau.to_csv(output_file, index=False)

    if verbose:
        print(f"Tableau-ready data saved to: {output_file}")

    return df_tableau


def calculate_metrics(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculate growth, averages, volatility and categories."""

    target_dir = _ensure_output_dir(output_dir)
    year_cols = [str(year) for year in range(1997, 2017)]

    df_analysis = df.copy()
    df_analysis["GDHI_1997"] = df["1997"]
    df_analysis["GDHI_2016"] = df["2016"]
    df_analysis["Absolute_Change"] = df["2016"] - df["1997"]
    df_analysis["Percentage_Growth"] = ((df["2016"] - df["1997"]) / df["1997"]) * 100
    df_analysis["Average_GDHI"] = df[year_cols].mean(axis=1)
    df_analysis["Volatility"] = df[year_cols].std(axis=1)

    df_analysis["Income_Category_2016"] = pd.cut(
        df["2016"],
        bins=[0, 15000, 20000, 25000, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    )
    df_analysis["Growth_Category"] = pd.cut(
        df_analysis["Percentage_Growth"],
        bins=[0, 60, 80, 100, np.inf],
        labels=["Slow", "Moderate", "Fast", "Very Fast"],
    )

    # Always save to CSV
    output_file = target_dir / "gdhi_with_metrics.csv"
    df_analysis.to_csv(output_file, index=False)

    if verbose:
        print(f"Analysis file with metrics saved to: {output_file}")

    return df_analysis


def generate_summary_statistics(df: pd.DataFrame, df_analysis: pd.DataFrame) -> None:
    """Print comprehensive summary statistics."""

    year_cols = [str(year) for year in range(1997, 2017)]

    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATA SUMMARY")
    print("=" * 80)

    print(f"\nTotal Regions: {len(df)}")
    print(f"Time Period: 1997-2016 ({len(year_cols)} years)")

    print("\n--- 1997 Statistics ---")
    print(f"Mean GDHI: GBP{df['1997'].mean():,.0f}")
    print(f"Median GDHI: GBP{df['1997'].median():,.0f}")
    print(f"Min GDHI: GBP{df['1997'].min():,.0f} ({df.loc[df['1997'].idxmin(), 'AREANM']})")
    print(f"Max GDHI: GBP{df['1997'].max():,.0f} ({df.loc[df['1997'].idxmax(), 'AREANM']})")
    print(f"Std Dev: GBP{df['1997'].std():,.0f}")

    print("\n--- 2016 Statistics ---")
    print(f"Mean GDHI: GBP{df['2016'].mean():,.0f}")
    print(f"Median GDHI: GBP{df['2016'].median():,.0f}")
    print(f"Min GDHI: GBP{df['2016'].min():,.0f} ({df.loc[df['2016'].idxmin(), 'AREANM']})")
    print(f"Max GDHI: GBP{df['2016'].max():,.0f} ({df.loc[df['2016'].idxmax(), 'AREANM']})")
    print(f"Std Dev: GBP{df['2016'].std():,.0f}")

    print("\n--- Growth Statistics (1997-2016) ---")
    print(f"Average Growth: {df_analysis['Percentage_Growth'].mean():.1f}%")
    print(f"Median Growth: {df_analysis['Percentage_Growth'].median():.1f}%")
    max_idx = df_analysis["Percentage_Growth"].idxmax()
    min_idx = df_analysis["Percentage_Growth"].idxmin()
    print(
        f"Highest Growth: {df_analysis.loc[max_idx, 'Percentage_Growth']:.1f}% "
        f"({df_analysis.loc[max_idx, 'AREANM']})"
    )
    print(
        f"Lowest Growth: {df_analysis.loc[min_idx, 'Percentage_Growth']:.1f}% "
        f"({df_analysis.loc[min_idx, 'AREANM']})"
    )

    print("\n--- Inequality Metrics ---")
    print(f"1997 Income Gap: GBP{df['1997'].max() - df['1997'].min():,.0f}")
    print(f"2016 Income Gap: GBP{df['2016'].max() - df['2016'].min():,.0f}")
    print(
        "Inequality Change: "
        f"GBP{(df['2016'].max() - df['2016'].min()) - (df['1997'].max() - df['1997'].min()):,.0f}"
    )

    print("\n--- Income Distribution (2016) ---")
    print(df_analysis["Income_Category_2016"].value_counts().sort_index())

    print("\n--- Growth Distribution ---")
    print(df_analysis["Growth_Category"].value_counts().sort_index())


def identify_interesting_regions(df_analysis: pd.DataFrame) -> None:
    """Print regions that stand out."""

    print("\n" + "=" * 80)
    print("REGIONS OF INTEREST FOR TABLEAU ANALYSIS")
    print("=" * 80)

    print("\n1. HIGHEST INCOME REGIONS (2016):")
    top_income = df_analysis.nlargest(5, "GDHI_2016")[["AREANM", "GDHI_2016", "Percentage_Growth"]]
    for _, row in top_income.iterrows():
        print(f"   - {row['AREANM']}: GBP{row['GDHI_2016']:,.0f} ({row['Percentage_Growth']:.1f}% growth)")

    print("\n2. FASTEST GROWING REGIONS:")
    fastest = df_analysis.nlargest(5, "Percentage_Growth")[["AREANM", "GDHI_1997", "GDHI_2016", "Percentage_Growth"]]
    for _, row in fastest.iterrows():
        print(f"   - {row['AREANM']}: {row['Percentage_Growth']:.1f}% (GBP{row['GDHI_1997']:,.0f} -> GBP{row['GDHI_2016']:,.0f})")

    print("\n3. SLOWEST GROWING REGIONS:")
    slowest = df_analysis.nsmallest(5, "Percentage_Growth")[["AREANM", "GDHI_1997", "GDHI_2016", "Percentage_Growth"]]
    for _, row in slowest.iterrows():
        print(f"   - {row['AREANM']}: {row['Percentage_Growth']:.1f}% (GBP{row['GDHI_1997']:,.0f} -> GBP{row['GDHI_2016']:,.0f})")

    print("\n4. REGIONS WITH HIGHEST VOLATILITY:")
    volatile = df_analysis.nlargest(5, "Volatility")[["AREANM", "Volatility", "Average_GDHI"]]
    for _, row in volatile.iterrows():
        print(f"   - {row['AREANM']}: Std Dev GBP{row['Volatility']:,.0f} (Avg: GBP{row['Average_GDHI']:,.0f})")


def create_comparison_groups(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Create region groupings for Tableau dashboards."""

    target_dir = _ensure_output_dir(output_dir)

    def classify_region(area_name: str) -> str:
        area_lower = str(area_name).lower()
        london_terms = [
            "westminster",
            "camden",
            "kensington",
            "wandsworth",
            "barnet",
            "lambeth",
            "hackney",
            "islington",
            "brent",
            "richmond",
            "hammersmith",
            "fulham",
            "hounslow",
        ]
        northern_terms = ["manchester", "liverpool", "leeds", "sheffield", "newcastle", "sunderland", "hull"]
        midlands_terms = ["birmingham", "nottingham", "derby", "leicester", "coventry"]
        south_west_terms = ["cornwall", "devon", "bristol", "bath", "somerset"]
        south_east_terms = ["kent", "surrey", "essex", "sussex", "hampshire"]
        scotland_terms = ["scotland", "edinburgh", "glasgow", "aberdeen", "dundee", "highlands", "fife"]
        wales_terms = ["wales", "cardiff", "swansea", "newport"]
        ni_terms = ["belfast", "derry", "armagh", "down", "antrim"]

        if any(term in area_lower for term in london_terms):
            return "London"
        if any(term in area_lower for term in northern_terms):
            return "Northern England"
        if any(term in area_lower for term in midlands_terms):
            return "Midlands"
        if any(term in area_lower for term in south_west_terms):
            return "South West"
        if any(term in area_lower for term in south_east_terms):
            return "South East"
        if any(term in area_lower for term in scotland_terms):
            return "Scotland"
        if any(term in area_lower for term in wales_terms):
            return "Wales"
        if any(term in area_lower for term in ni_terms):
            return "Northern Ireland"
        return "Other"

    df_groups = df.copy()
    df_groups["Region_Group"] = df_groups["AREANM"].apply(classify_region)

    # Always save to CSV
    output_file = target_dir / "gdhi_with_regions.csv"
    df_groups.to_csv(output_file, index=False)

    if verbose:
        print(f"Regional groupings saved to: {output_file}")
        print("\n--- Regional Distribution ---")
        print(df_groups["Region_Group"].value_counts())

    return df_groups


def create_dashboard_figure(df: pd.DataFrame, df_analysis: pd.DataFrame) -> Figure:
    """Compose a dashboard figure for the GUI clients."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_dist, ax_growth, ax_income_cat, ax_trend = axes.flatten()

    ax_dist.hist(df["2016"], bins=25, color="#4472C4", edgecolor="black", alpha=0.7)
    ax_dist.set_title("GDHI Distribution (2016)", fontweight="bold")
    ax_dist.set_xlabel("GDHI (GBP)", fontweight="bold")
    ax_dist.set_ylabel("Number of areas", fontweight="bold")
    ax_dist.grid(True, alpha=0.3)

    growth_sorted = df_analysis.sort_values("Percentage_Growth", ascending=True).tail(15)
    colors = ["#2E7D32" if x > 80 else "#C62828" for x in growth_sorted["Percentage_Growth"]]
    ax_growth.barh(growth_sorted["AREANM"], growth_sorted["Percentage_Growth"], color=colors)
    ax_growth.set_title("Percentage Growth (1997-2016)", fontweight="bold")
    ax_growth.set_xlabel("Growth (%)", fontweight="bold")
    ax_growth.tick_params(axis="y", labelsize=8)

    income_counts = df_analysis["Income_Category_2016"].value_counts().sort_index()
    ax_income_cat.bar(income_counts.index.astype(str), income_counts.values, color="#1F77B4")
    ax_income_cat.set_title("Income Categories (2016)", fontweight="bold")
    ax_income_cat.set_xlabel("Category", fontweight="bold")
    ax_income_cat.set_ylabel("Regions", fontweight="bold")
    ax_income_cat.grid(axis="y", alpha=0.3)

    year_cols = [str(year) for year in range(1997, 2017)]
    uk_avg = df[year_cols].mean()
    ax_trend.plot(year_cols, uk_avg, marker="o", linewidth=2, color="#E15759")
    ax_trend.set_title("UK Average GDHI Over Time", fontweight="bold")
    ax_trend.set_xlabel("Year", fontweight="bold")
    ax_trend.set_ylabel("Average GDHI (GBP)", fontweight="bold")
    ax_trend.tick_params(axis="x", rotation=45)
    ax_trend.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    df, actual_path = load_data(config.get("data_path"), verbose=False)
    _ = create_tableau_ready_data(df, output_dir=Path("outputs"), verbose=False)
    df_metrics = calculate_metrics(df, output_dir=Path("outputs"), verbose=False)
    fig = create_dashboard_figure(df, df_metrics)

    st.header("Week 8 - GDHI data preparation for Tableau")
    st.caption(f"Data source: {actual_path}")
    st.pyplot(fig)


def build_widget(config: dict):
    """Build widget for Qt GUI - returns dict with figures and text."""
    try:
        df, actual_path = load_data(config.get("data_path"), verbose=False)
        _ = create_tableau_ready_data(df, output_dir=Path("outputs"), verbose=False)
        df_metrics = calculate_metrics(df, output_dir=Path("outputs"), verbose=False)
        fig = create_dashboard_figure(df, df_metrics)

        # Return standard format: dict with figures and text
        return {
            "figures": [("GDHI Tableau Dashboard", fig)],
            "text": [("Info", "Week 8: Data prepared for Tableau visualization\n\n"
                     f"Data source: {actual_path}\n"
                     "Outputs created in ./outputs/ directory:\n"
                     "  • gdhi_tableau_format.csv\n"
                     "  • gdhi_with_metrics.csv\n"
                     "  • gdhi_with_regions.csv")]
        }
    except Exception as exc:
        return {
            "text": [("Error", f"Week 8 error: {exc}")]
        }


def main() -> None:
    print("=" * 80)
    print("GDHI DATA PREPARATION FOR TABLEAU")
    print("Week 8 - Analytics and Business Intelligence")
    print("=" * 80)

    df, _ = load_data(None, verbose=True)
    df_tableau = create_tableau_ready_data(df, output_dir=Path("outputs"), verbose=True)
    df_metrics = calculate_metrics(df, output_dir=Path("outputs"), verbose=True)
    generate_summary_statistics(df, df_metrics)
    identify_interesting_regions(df_metrics)
    create_comparison_groups(df, output_dir=Path("outputs"), verbose=True)

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print("\nFiles created (in ./outputs):")
    print("1. gdhi_tableau_format.csv - Time-series visuals")
    print("2. gdhi_with_metrics.csv - Growth and comparison analysis")
    print("3. gdhi_with_regions.csv - Regional grouping analysis")

    create_dashboard_figure(df, df_metrics)
    plt.show(block=False)


if __name__ == "__main__":
    main()
