from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from utils.data_loader import default_csv_path, load_csv
from utils.qt_mpl import MplWidget  # type: ignore

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def _ensure_output_dir(output_dir: Path | None) -> Path:
    target = output_dir or Path("outputs")
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_and_explore_data(path: str | None, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Load GDHI data and optionally print the exploratory summary."""

    actual_path = path or str(default_csv_path())
    df = load_csv(actual_path).copy()

    if verbose:
        print("=" * 80)
        print("GDHI DATASET EXPLORATION")
        print("=" * 80)
        print("\n1. DATASET OVERVIEW")
        print("-" * 80)
        print(f"Shape: {df.shape[0]} areas × {df.shape[1]} columns")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df.columns[2]} to {df.columns[-1]}")

        print("\n2. SAMPLE DATA (First 5 rows)")
        print("-" * 80)
        print(df.head())

        print("\n3. DATA TYPES")
        print("-" * 80)
        print(df.dtypes)

        print("\n4. MISSING VALUES")
        print("-" * 80)
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("OK No missing values found")
        else:
            print(missing[missing > 0])

    return df, actual_path


def calculate_statistics(df: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """Calculate key statistics for 2016."""

    stats_2016 = df["2016"].describe()
    if verbose:
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS FOR 2016")
        print("=" * 80)
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 80)
        print(stats_2016)

        print("\n2. KEY METRICS")
        print("-" * 80)
        print(f"UK Average Income (2016): GBP{df['2016'].mean():,.0f}")
        print(f"Median Income (2016): GBP{df['2016'].median():,.0f}")
        print(f"Income Range: GBP{df['2016'].max() - df['2016'].min():,.0f}")
        print(f"Standard Deviation: GBP{df['2016'].std():,.0f}")
        cv = (df["2016"].std() / df["2016"].mean()) * 100
        print(f"Coefficient of Variation: {cv:.2f}%")

        print("\n3. INCOME QUARTILES (2016)")
        print("-" * 80)
        q1 = df["2016"].quantile(0.25)
        q2 = df["2016"].quantile(0.50)
        q3 = df["2016"].quantile(0.75)
        print(f"Q1 (25th percentile): GBP{q1:,.0f}")
        print(f"Q2 (50th percentile): GBP{q2:,.0f}")
        print(f"Q3 (75th percentile): GBP{q3:,.0f}")
        print(f"Interquartile Range: GBP{q3 - q1:,.0f}")

        print("\n4. TOP 10 HIGHEST INCOME AREAS (2016)")
        print("-" * 80)
        top_10 = df.nlargest(10, "2016")[["AREANM", "2016"]]
        for _, row in top_10.iterrows():
            print(f"{row['AREANM']:.<50} GBP{row['2016']:>7,}")

        print("\n5. BOTTOM 10 LOWEST INCOME AREAS (2016)")
        print("-" * 80)
        bottom_10 = df.nsmallest(10, "2016")[["AREANM", "2016"]]
        for _, row in bottom_10.iterrows():
            print(f"{row['AREANM']:.<50} GBP{row['2016']:>7,}")

        print("\n6. INEQUALITY METRICS")
        print("-" * 80)
        max_income = df["2016"].max()
        min_income = df["2016"].min()
        print(f"Highest Income Area: GBP{max_income:,}")
        print(f"Lowest Income Area: GBP{min_income:,}")
        print(f"Income Gap: GBP{max_income - min_income:,}")
        print(f"Ratio (Max/Min): {max_income / min_income:.2f}x")

    return stats_2016


def analyze_growth(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Analyze income growth from 1997 to 2016."""

    df_growth = df.copy()
    df_growth["Total_Growth"] = ((df["2016"] - df["1997"]) / df["1997"]) * 100
    df_growth["Absolute_Change"] = df["2016"] - df["1997"]
    df_growth["Avg_Annual_Growth"] = ((df["2016"] / df["1997"]) ** (1 / 19) - 1) * 100

    if verbose:
        print("\n" + "=" * 80)
        print("GROWTH ANALYSIS (1997-2016)")
        print("=" * 80)
        print("\n1. OVERALL GROWTH STATISTICS")
        print("-" * 80)
        print(f"Average Total Growth: {df_growth['Total_Growth'].mean():.2f}%")
        print(f"Median Total Growth: {df_growth['Total_Growth'].median():.2f}%")
        print(f"Average Annual Growth Rate: {df_growth['Avg_Annual_Growth'].mean():.2f}%")

        print("\n2. TOP 10 FASTEST GROWING AREAS (1997-2016)")
        print("-" * 80)
        top_growth = df_growth.nlargest(10, "Total_Growth")[["AREANM", "1997", "2016", "Total_Growth"]]
        for _, row in top_growth.iterrows():
            print(
                f"{row['AREANM']:.<40} GBP{row['1997']:>6,} -> GBP{row['2016']:>6,} "
                f"({row['Total_Growth']:>6.2f}%)"
            )

        print("\n3. BOTTOM 10 SLOWEST GROWING AREAS (1997-2016)")
        print("-" * 80)
        slow_growth = df_growth.nsmallest(10, "Total_Growth")[["AREANM", "1997", "2016", "Total_Growth"]]
        for _, row in slow_growth.iterrows():
            print(
                f"{row['AREANM']:.<40} GBP{row['1997']:>6,} -> GBP{row['2016']:>6,} "
                f"({row['Total_Growth']:>6.2f}%)"
            )

        print("\n4. UK AVERAGE INCOME GROWTH OVER TIME")
        print("-" * 80)
        year_cols = [str(year) for year in range(1997, 2017)]
        uk_avg = df[year_cols].mean()
        for year in [1997, 2000, 2005, 2010, 2015, 2016]:
            print(f"{year}: GBP{uk_avg[str(year)]:,.0f}")
        total_growth = ((uk_avg["2016"] - uk_avg["1997"]) / uk_avg["1997"]) * 100
        print(f"\nTotal UK Average Growth: {total_growth:.2f}%")

    return df_growth


def prepare_for_tableau(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare and save datasets formatted for Tableau dashboards."""
    
    target_dir = _ensure_output_dir(output_dir)

    year_cols = [str(year) for year in range(1997, 2017)]
    df_long = pd.melt(
        df,
        id_vars=["AREANM", "AREACD"],
        value_vars=year_cols,
        var_name="Year",
        value_name="GDHI",
    )
    df_long["Year"] = df_long["Year"].astype(int)

    uk_avg = df_long.groupby("Year")["GDHI"].mean().reset_index().rename(columns={"GDHI": "UK_Average"})
    df_long = df_long.merge(uk_avg, on="Year", how="left")
    df_long["Diff_from_UK_Avg"] = df_long["GDHI"] - df_long["UK_Average"]
    df_long["Pct_of_UK_Avg"] = (df_long["GDHI"] / df_long["UK_Average"]) * 100
    df_long["Income_Category"] = pd.cut(
        df_long["GDHI"],
        bins=[0, 15000, 18000, 22000, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    )

    df_growth = df.copy()
    df_growth["Growth_1997_2016_Pct"] = ((df["2016"] - df["1997"]) / df["1997"]) * 100
    df_growth["Growth_1997_2016_Abs"] = df["2016"] - df["1997"]
    df_growth["Growth_2007_2016_Pct"] = ((df["2016"] - df["2007"]) / df["2007"]) * 100
    df_growth["Avg_Annual_Growth"] = ((df["2016"] / df["1997"]) ** (1 / 19) - 1) * 100
    df_growth["Growth_Category"] = pd.cut(
        df_growth["Growth_1997_2016_Pct"],
        bins=[0, 80, 100, 120, np.inf],
        labels=["Slow", "Moderate", "Fast", "Very Fast"],
    )

    df_quartiles = df.copy()
    df_quartiles["Q1997"] = pd.qcut(df["1997"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    df_quartiles["Q2016"] = pd.qcut(df["2016"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    df_quartiles["Quartile_Change"] = (
        df_quartiles["Q2016"].astype(str) + " from " + df_quartiles["Q1997"].astype(str)
    )

    if verbose:
        print("\n" + "=" * 80)
        print("PREPARING DATA FOR TABLEAU")
        print("=" * 80)
        print(f"\n1. Long format created: {df_long.shape}")
        print(df_long.head())
        print("\n2. Growth metrics calculated.")
        print("3. Quartile rankings set.")

    outputs = {
        "GDHI_Long_Format.csv": df_long,
        "GDHI_Growth_Metrics.csv": df_growth,
        "GDHI_Quartile_Analysis.csv": df_quartiles,
    }
    for filename, dataframe in outputs.items():
        filepath = target_dir / filename
        dataframe.to_csv(filepath, index=False)
        if verbose:
            print(f"   OK Saved: {filepath}")

    return df_long, df_growth, df_quartiles


def generate_insights(df: pd.DataFrame, df_growth: pd.DataFrame, verbose: bool = True) -> None:
    """Print narrative insights to complement the dashboard."""

    if not verbose:
        return

    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR YOUR DASHBOARD")
    print("=" * 80)

    top_area = df.nlargest(1, "2016")[["AREANM", "2016"]].iloc[0]
    bottom_area = df.nsmallest(1, "2016")[["AREANM", "2016"]].iloc[0]
    print("\n[DATA] INSIGHT 1: EXTREME REGIONAL INEQUALITY")
    print("-" * 80)
    ratio = top_area["2016"] / bottom_area["2016"]
    print(f"The wealthiest area ({top_area['AREANM']}) has {ratio:.2f}x more disposable income "
          f"than the poorest area ({bottom_area['AREANM']}).")

    print("\n🏙️ INSIGHT 2: LONDON DOMINANCE")
    print("-" * 80)
    london_count = df[df["2016"] > 30000].shape[0]
    top_decile = df[df["2016"] > df["2016"].quantile(0.9)].shape[0]
    print(f"All {london_count} areas with income above GBP30,000 are in London.")
    print(f"London regions account for {top_decile} of the top 10% highest-income areas.")

    print("\n[GROWTH] INSIGHT 3: DIVERGENT GROWTH PATTERNS")
    print("-" * 80)
    avg_growth = df_growth["Total_Growth"].mean()
    above_avg = (df_growth["Total_Growth"] > avg_growth).sum()
    print(f"Average growth: {avg_growth:.1f}%")
    print(f"{above_avg} areas ({above_avg/len(df_growth)*100:.1f}%) grew faster than average.")

    print("\n📉 INSIGHT 4: 2008 FINANCIAL CRISIS IMPACT")
    print("-" * 80)
    growth_07_08 = ((df["2008"].mean() - df["2007"].mean()) / df["2007"].mean()) * 100
    growth_08_09 = ((df["2009"].mean() - df["2008"].mean()) / df["2008"].mean()) * 100
    trend = "slowdown" if growth_08_09 < growth_07_08 else "acceleration"
    print(f"Average income growth 2007-2008: {growth_07_08:.2f}%")
    print(f"Average income growth 2008-2009: {growth_08_09:.2f}% "
          f"(indicating a {trend} in income growth).")


def create_dashboard_figure(df: pd.DataFrame, df_growth: pd.DataFrame) -> Figure:
    """Construct a four-panel summary dashboard for the Qt/Streamlit clients."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_dist, ax_bar, ax_growth, ax_line = axes.flatten()

    data_2016 = df["2016"]
    ax_dist.hist(data_2016, bins=25, color="#4472C4", edgecolor="black", alpha=0.7)
    ax_dist.axvline(data_2016.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: GBP{data_2016.mean():,.0f}")
    ax_dist.set_title("GDHI Distribution (2016)", fontweight="bold")
    ax_dist.set_xlabel("GDHI (GBP)", fontweight="bold")
    ax_dist.set_ylabel("Number of areas", fontweight="bold")
    ax_dist.legend(fontsize=9)
    ax_dist.grid(True, alpha=0.3)

    top_regions = df.nlargest(10, "2016")[["AREANM", "2016"]].iloc[::-1]
    ax_bar.barh(top_regions["AREANM"], top_regions["2016"], color="#2E7D32")
    ax_bar.set_title("Top 10 Regions by GDHI (2016)", fontweight="bold")
    ax_bar.set_xlabel("GDHI (GBP)", fontweight="bold")
    ax_bar.tick_params(axis="y", labelsize=8)

    growth_sorted = df_growth.sort_values("Total_Growth", ascending=True).tail(15)
    colors = ["#2E7D32" if x > 100 else "#C62828" for x in growth_sorted["Total_Growth"]]
    ax_growth.barh(growth_sorted["AREANM"], growth_sorted["Total_Growth"], color=colors)
    ax_growth.set_title("Growth 1997-2016 (%)", fontweight="bold")
    ax_growth.set_xlabel("Total growth (%)", fontweight="bold")
    ax_growth.tick_params(axis="y", labelsize=8)

    year_cols = [str(year) for year in range(1997, 2017)]
    uk_avg = df[year_cols].mean()
    ax_line.plot(year_cols, uk_avg, marker="o", linewidth=2, color="#1F77B4")
    ax_line.set_title("UK Average GDHI Over Time", fontweight="bold")
    ax_line.set_xlabel("Year", fontweight="bold")
    ax_line.set_ylabel("Average GDHI (GBP)", fontweight="bold")
    ax_line.tick_params(axis="x", rotation=45)
    ax_line.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    df, actual_path = load_and_explore_data(config.get("data_path"), verbose=False)
    df_growth = analyze_growth(df, verbose=False)
    fig = create_dashboard_figure(df, df_growth)

    st.header("Week 10 - GDHI data exploration")
    st.caption(f"Data source: {actual_path}")
    st.pyplot(fig)


def build_widget(config: dict):
    """Build widget for Qt GUI - returns dict with figures and text."""
    try:
        df, actual_path = load_and_explore_data(config.get("data_path"), verbose=False)
        df_growth = analyze_growth(df, verbose=False)
        # Generate CSV files for Tableau
        prepare_for_tableau(df, output_dir=Path("outputs"), verbose=False)
        fig = create_dashboard_figure(df, df_growth)

        # Generate insights text
        insights_text = (
            "Week 10: GDHI Data Exploration for Tableau\n\n"
            f"Data source: {actual_path}\n\n"
            "KEY INSIGHTS:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "[DATA] Regional Inequality:\n"
            f"   - Dataset covers {len(df)} UK regions (1997-2016)\n"
            f"   - Wealthiest area: {df.nlargest(1, '2016')['AREANM'].iloc[0]} (GBP{df['2016'].max():,.0f})\n"
            f"   - Poorest area: {df.nsmallest(1, '2016')['AREANM'].iloc[0]} (GBP{df['2016'].min():,.0f})\n"
            f"   - Income ratio: {df['2016'].max() / df['2016'].min():.2f}x difference\n\n"
            "[GROWTH] Growth Analysis:\n"
            f"   - Average growth 1997-2016: {df_growth['Total_Growth'].mean():.1f}%\n"
            f"   - Fastest growing: {df_growth.nlargest(1, 'Total_Growth')['AREANM'].iloc[0]} "
            f"({df_growth['Total_Growth'].max():.1f}%)\n"
            f"   - Slowest growing: {df_growth.nsmallest(1, 'Total_Growth')['AREANM'].iloc[0]} "
            f"({df_growth['Total_Growth'].min():.1f}%)\n\n"
            "[INFO] Dashboard Visualizations:\n"
            "   - Income distribution (2016)\n"
            "   - Top 10 regions by income\n"
            "   - Regional growth comparison\n"
            "   - UK average income trend over time"
        )

        # Return standard format: dict with figures and text
        return {
            "figures": [("GDHI Exploration Dashboard", fig)],
            "text": [("Analysis Summary", insights_text)]
        }
    except Exception as exc:
        import traceback
        return {
            "text": [("Error", f"Week 10 error: {exc}\n\n{traceback.format_exc()}")]
        }


def main() -> None:
    print("\n" + "=" * 80)
    print("WEEK 10: GDHI DATA ANALYSIS FOR TABLEAU DASHBOARDS")
    print("=" * 80)

    df, actual_path = load_and_explore_data(None, verbose=True)
    calculate_statistics(df, verbose=True)
    df_growth = analyze_growth(df, verbose=True)
    prepare_for_tableau(df, output_dir=Path("outputs"), verbose=True)
    generate_insights(df, df_growth, verbose=True)

    print("\nOutput directory: outputs/")
    print("Files ready for Tableau:")
    for filename in ["GDHI_Long_Format.csv", "GDHI_Growth_Metrics.csv", "GDHI_Quartile_Analysis.csv"]:
        print(f"  • outputs/{filename}")

    create_dashboard_figure(df, df_growth)
    plt.show(block=False)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
