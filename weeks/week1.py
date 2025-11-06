"""Week 1 - GDHI analysis helpers for the Qt desktop app."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from utils.data_loader import default_csv_path, load_csv

YEAR_COLUMNS = [str(year) for year in range(1997, 2017)]
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(filepath: str | None):
    actual_path = filepath or str(default_csv_path())
    df = load_csv(actual_path)
    print(f"✓ Dataset loaded from {actual_path}: {df.shape[0]} regions, {df.shape[1]} columns")
    return df, actual_path


def basic_statistics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["Growth_GBP"] = df["2016"] - df["1997"]
    df["Growth_pct"] = ((df["2016"] - df["1997"]) / df["1997"]) * 100

    if verbose:
        print("\n" + "=" * 80)
        print("BASIC STATISTICS")
        print("=" * 80)

        print("\n1997 Statistics:")
        print(f"   Mean: GBP {df['1997'].mean():.2f}")
        print(f"   Median: GBP {df['1997'].median():.2f}")
        print(f"   Std Dev: GBP {df['1997'].std():.2f}")
        print(f"   Range: GBP {df['1997'].min():.2f} - GBP {df['1997'].max():.2f}")

        print("\n2016 Statistics:")
        print(f"   Mean: GBP {df['2016'].mean():.2f}")
        print(f"   Median: GBP {df['2016'].median():.2f}")
        print(f"   Std Dev: GBP {df['2016'].std():.2f}")
        print(f"   Range: GBP {df['2016'].min():.2f} - GBP {df['2016'].max():.2f}")

        print("\nOverall Growth:")
        print(f"   Average increase: GBP {df['Growth_GBP'].mean():.2f}")
        print(f"   Percentage increase: {df['Growth_pct'].mean():.1f}%")

    return df


def regional_analysis(df: pd.DataFrame, n: int = 10, verbose: bool = True) -> None:
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("REGIONAL ANALYSIS")
    print("=" * 80)

    print(f"\nTop {n} Regions (2016):")
    top_regions = df.nlargest(n, "2016")[["AREANM", "2016", "Growth_pct"]]
    for _, row in top_regions.iterrows():
        print(f"   {row['AREANM']:<50} GBP {row['2016']:>7,.0f} (+{row['Growth_pct']:>5.1f}%)")

    print(f"\nBottom {n} Regions (2016):")
    bottom_regions = df.nsmallest(n, "2016")[["AREANM", "2016", "Growth_pct"]]
    for _, row in bottom_regions.iterrows():
        print(f"   {row['AREANM']:<50} GBP {row['2016']:>7,.0f} (+{row['Growth_pct']:>5.1f}%)")

    print("\nFastest Growing Regions:")
    growth_leaders = df.nlargest(n, "Growth_pct")[["AREANM", "1997", "2016", "Growth_pct"]]
    for _, row in growth_leaders.iterrows():
        print(f"   {row['AREANM']:<50} {row['Growth_pct']:>5.1f}% (GBP {row['1997']:,} -> GBP {row['2016']:,})")

    print("\nRegional Inequality:")
    print(f"   Highest: GBP {df['2016'].max():,} ({df.loc[df['2016'].idxmax(), 'AREANM']})")
    print(f"   Lowest: GBP {df['2016'].min():,} ({df.loc[df['2016'].idxmin(), 'AREANM']})")
    print(f"   Ratio: {df['2016'].max() / df['2016'].min():.2f}x")


def temporal_analysis(df: pd.DataFrame, verbose: bool = True) -> None:
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS")
    print("=" * 80)

    years = YEAR_COLUMNS
    avg_by_year = {year: df[year].mean() for year in years}

    print("\nYearly Averages:")
    for year in years:
        print(f"   {year}: GBP {avg_by_year[year]:>8,.2f}")

    print("\nYear-on-Year Growth Rates:")
    for i in range(1, len(years)):
        prev_year = years[i - 1]
        curr_year = years[i]
        growth = ((avg_by_year[curr_year] - avg_by_year[prev_year]) / avg_by_year[prev_year]) * 100
        print(f"   {prev_year} -> {curr_year}: {growth:>5.2f}%")


def create_visualizations(df: pd.DataFrame) -> Figure:
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(df["1997"], bins=30, alpha=0.5, label="1997", color="steelblue")
    ax1.hist(df["2016"], bins=30, alpha=0.5, label="2016", color="coral")
    ax1.set_xlabel("GDHI per head (GBP)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of GDHI: 1997 vs 2016")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    top_10 = df.nlargest(10, "2016").sort_values("2016")
    ax2.barh(range(len(top_10)), top_10["2016"], color="green", alpha=0.7)
    ax2.set_yticks(range(len(top_10)))
    ax2.set_yticklabels([name[:30] for name in top_10["AREANM"]], fontsize=8)
    ax2.set_xlabel("GDHI per head (GBP)")
    ax2.set_title("Top 10 Regions (2016)")
    ax2.grid(True, alpha=0.3, axis="x")

    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(df["Growth_pct"], bins=30, color="purple", alpha=0.7, edgecolor="black")
    ax3.axvline(df["Growth_pct"].mean(), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {df['Growth_pct'].mean():.1f}%")
    ax3.set_xlabel("Growth Rate (%)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Growth Rate Distribution (1997-2016)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    avg_gdhi = [df[year].mean() for year in YEAR_COLUMNS]
    years_numeric = [int(year) for year in YEAR_COLUMNS]
    ax4.plot(years_numeric, avg_gdhi, marker="o", linewidth=2, markersize=4, color="darkblue")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Average GDHI per head (GBP)")
    ax4.set_title("National Average GDHI Trend (1997-2016)")
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(years_numeric)
    ax4.set_xticklabels(YEAR_COLUMNS, rotation=45)
    ax4.axvspan(2007, 2009, alpha=0.2, color="red", label="Financial Crisis")
    ax4.legend()

    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(df["1997"], df["2016"], alpha=0.6, s=50, color="teal")
    min_val = min(df["1997"].min(), df["2016"].min())
    max_val = max(df["1997"].max(), df["2016"].max())
    ax5.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="No change")
    ax5.set_xlabel("GDHI 1997 (GBP)")
    ax5.set_ylabel("GDHI 2016 (GBP)")
    ax5.set_title("GDHI: 1997 vs 2016")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = plt.subplot(2, 3, 6)
    decade_data = {
        "Late 90s\n(97-99)": df[["1997", "1998", "1999"]].mean(axis=1),
        "Early 00s\n(00-04)": df[["2000", "2001", "2002", "2003", "2004"]].mean(axis=1),
        "Late 00s\n(05-09)": df[["2005", "2006", "2007", "2008", "2009"]].mean(axis=1),
        "Early 10s\n(10-14)": df[["2010", "2011", "2012", "2013", "2014"]].mean(axis=1),
        "Mid 10s\n(15-16)": df[["2015", "2016"]].mean(axis=1),
    }
    decade_series = list(decade_data.values())
    ax6.boxplot(decade_series)
    ax6.set_xticks(range(1, len(decade_series) + 1))
    ax6.set_xticklabels(decade_data.keys(), rotation=15)
    ax6.set_ylabel("Average GDHI per head (GBP)")
    ax6.set_title("GDHI Distribution by Period")
    ax6.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def export_summary(df: pd.DataFrame, output_path: Path, verbose: bool = True) -> None:
    summary = pd.DataFrame(
        {
            "Region": df["AREANM"],
            "Code": df["AREACD"],
            "GDHI_1997": df["1997"],
            "GDHI_2016": df["2016"],
            "Absolute_Growth": df["Growth_GBP"],
            "Percentage_Growth": df["Growth_pct"],
        }
    ).sort_values("GDHI_2016", ascending=False)
    summary.to_csv(output_path, index=False)
    if verbose:
        print(f"✓ Summary exported to: {output_path}")


def prepare_outputs(path: str | None, verbose: bool = True) -> Tuple[List[Tuple[str, Figure]], str, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df, actual_path = load_data(path)
        df = basic_statistics(df, verbose=True)
        regional_analysis(df, verbose=True)
        temporal_analysis(df, verbose=True)
        fig = create_visualizations(df)
        export_summary(df, OUTPUT_DIR / "week1_gdhi_summary.csv", verbose=True)
    summary = buffer.getvalue()
    if verbose:
        print(summary)
    figures = [("Executive Overview", fig)]
    fig.savefig(OUTPUT_DIR / "week1_executive_overview.png", dpi=300, bbox_inches="tight")
    return figures, summary, actual_path


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    figures, summary, actual_path = prepare_outputs(config.get("data_path"), verbose=False)
    st.header("Week 1 - GDHI data analysis")
    st.caption(f"Data source: {actual_path}")
    label, figure = figures[0]
    st.pyplot(figure)
    st.markdown("**Summary**")
    st.code(summary, language="text")


def build_widget(config: dict):
    figures, summary, _ = prepare_outputs(config.get("data_path"), verbose=False)
    return {"figures": figures, "text": [("Summary", summary)]}


def main() -> None:
    prepare_outputs(None, verbose=True)
    plt.show(block=False)
    print("\nAnalysis complete. Outputs written to the outputs/ directory.")


if __name__ == "__main__":
    main()
