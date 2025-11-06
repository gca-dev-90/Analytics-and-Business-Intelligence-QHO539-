"""Week 9 - Tableau-equivalent visualizations for GDHI data analysis."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from matplotlib.figure import Figure

from utils.data_loader import default_csv_path, load_csv

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: str | None, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Load the GDHI dataset and create calculated fields."""
    actual_path = path or str(default_csv_path())
    df = load_csv(actual_path).copy()

    if verbose:
        print("=" * 80)
        print("WEEK 9: TABLEAU-EQUIVALENT VISUALIZATIONS")
        print("=" * 80)
        print(f"\n[OK] Data loaded from {actual_path}")
        print(f"Rows: {len(df)}  Columns: {len(df.columns)}")

    # Create calculated fields
    year_cols = [str(year) for year in range(1997, 2017)]

    # Growth Rate 1997-2016
    df['Growth_Rate_1997_to_2016'] = ((df['2016'] - df['1997']) / df['1997']) * 100

    # Recession Impact (2008-2009)
    df['Recession_Change'] = df['2009'] - df['2008']
    df['Recession_Change_Percent'] = ((df['2009'] - df['2008']) / df['2008']) * 100

    # Recovery Rate (2009-2016)
    df['Recovery_2009_to_2016'] = df['2016'] - df['2009']
    df['Recovery_Rate_Percent'] = ((df['2016'] - df['2009']) / df['2009']) * 100

    # Deviation from Average (2016)
    avg_2016 = df['2016'].mean()
    df['Deviation_From_Avg_2016'] = df['2016'] - avg_2016

    # Income Range (Volatility measure)
    df['Income_Range_1997_2016'] = df[year_cols].max(axis=1) - df[year_cols].min(axis=1)

    # Income Category (2016)
    df['Income_Category_2016'] = pd.cut(
        df['2016'],
        bins=[0, 15000, 20000, float('inf')],
        labels=['Low Income', 'Medium Income', 'High Income']
    )

    # Region Group (based on NUTS codes)
    df['Area_Code_Prefix'] = df['AREACD'].str[:3]
    region_mapping = {
        'UKC': 'North East',
        'UKD': 'North West',
        'UKE': 'Yorkshire',
        'UKF': 'East Midlands',
        'UKG': 'West Midlands',
        'UKH': 'East of England',
        'UKI': 'London',
        'UKJ': 'South East',
        'UKK': 'South West',
        'UKL': 'Wales',
        'UKM': 'Scotland',
        'UKN': 'Northern Ireland'
    }
    df['Region_Group'] = df['Area_Code_Prefix'].map(region_mapping)
    df['Region_Group'].fillna('Unknown', inplace=True)

    # Performance Label
    conditions = [
        df['Growth_Rate_1997_to_2016'] > 100,
        df['Growth_Rate_1997_to_2016'] > 80
    ]
    choices = ['Strong Growth', 'Moderate Growth']
    df['Performance_Label'] = np.select(conditions, choices, default='Slow Growth')

    if verbose:
        print(f"Created calculated fields: Growth rates, categories, and regional groupings")

    return df, actual_path


def create_top_20_wealthiest_viz(df: pd.DataFrame, verbose: bool = True) -> Figure:
    """Visualization 1: Top 20 Wealthiest Areas by 2016 GDHI."""
    fig, ax = plt.subplots(figsize=(12, 10))

    top_20 = df.nlargest(20, '2016').sort_values('2016', ascending=True)

    colors = top_20['Income_Category_2016'].map({
        'High Income': '#2ecc71',
        'Medium Income': '#f39c12',
        'Low Income': '#e74c3c'
    })

    ax.barh(top_20['AREANM'], top_20['2016'], color=colors)
    ax.set_xlabel('GDHI per Head (GBP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Area Name', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Highest Income Areas (2016)', fontsize=14, fontweight='bold', pad=20)

    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax.text(row['2016'] + 200, i, f"GBP {row['2016']:,.0f}", va='center', fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='High Income (GBP 20,000)'),
        Patch(facecolor='#f39c12', label='Medium Income (GBP 15,000-GBP 20,000)'),
        Patch(facecolor='#e74c3c', label='Low Income (<GBP 15,000)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if verbose:
        fig.savefig(OUTPUT_DIR / "week9_top_20_wealthiest.png", dpi=300, bbox_inches='tight')
        print("[OK] Saved: week9_top_20_wealthiest.png")

    return fig


def create_growth_rate_viz(df: pd.DataFrame, verbose: bool = True) -> Figure:
    """Visualization 2: Growth Rate Analysis (1997-2016)."""
    fig, ax = plt.subplots(figsize=(12, 10))

    top_30 = df.nlargest(30, 'Growth_Rate_1997_to_2016').sort_values(
        'Growth_Rate_1997_to_2016', ascending=True
    )

    colors = cm.get_cmap("RdYlGn")(
        (top_30['Growth_Rate_1997_to_2016'] - top_30['Growth_Rate_1997_to_2016'].min()) /
        (top_30['Growth_Rate_1997_to_2016'].max() - top_30['Growth_Rate_1997_to_2016'].min())
    )

    ax.barh(top_30['AREANM'], top_30['Growth_Rate_1997_to_2016'], color=colors)
    ax.set_xlabel('Growth Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Area Name', fontsize=12, fontweight='bold')
    ax.set_title('Top 30 Areas by Growth Rate (1997-2016)', fontsize=14, fontweight='bold', pad=20)

    for i, (idx, row) in enumerate(top_30.iterrows()):
        ax.text(row['Growth_Rate_1997_to_2016'] + 1, i,
               f"{row['Growth_Rate_1997_to_2016']:.1f}%", va='center', fontsize=9)

    ax.axvline(x=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
              label='100% Growth (Doubled)')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if verbose:
        fig.savefig(OUTPUT_DIR / "week9_growth_rate_analysis.png", dpi=300, bbox_inches='tight')
        print("[OK] Saved: week9_growth_rate_analysis.png")

    return fig


def create_regional_comparison_viz(df: pd.DataFrame, verbose: bool = True) -> Figure:
    """Visualization 3: Regional Average Income Comparison."""
    fig, ax = plt.subplots(figsize=(12, 8))

    regional_avg = (
        df.groupby("Region_Group")["2016"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Average_GDHI", "count": "Count"})
        .sort_values("Average_GDHI", ascending=True)
    )

    if "Unknown" in regional_avg.index:
        regional_avg = regional_avg.drop("Unknown")

    colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(regional_avg)))
    avg_values = regional_avg["Average_GDHI"].to_numpy(dtype=float)
    ax.barh(regional_avg.index, avg_values, color=colors)

    for i, (_, row) in enumerate(regional_avg.iterrows()):
        avg_val = float(row["Average_GDHI"])
        ax.text(
            avg_val + 200,
            i,
            f"GBP {avg_val:,.0f}\n({int(row['Count'])} areas)",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Average GDHI per Head (GBP)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Region", fontsize=12, fontweight="bold")
    ax.set_title("Regional Income Comparison (2016)", fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if verbose:
        fig.savefig(OUTPUT_DIR / "week9_regional_comparison.png", dpi=300, bbox_inches='tight')
        print("[OK] Saved: week9_regional_comparison.png")

    return fig


def create_recession_impact_viz(df: pd.DataFrame, verbose: bool = True) -> Figure:
    """Visualization 4: Recession Impact Analysis (2008-2009)."""
    fig, ax = plt.subplots(figsize=(12, 10))

    top_impact = df.nlargest(15, 'Recession_Change', keep='first')
    bottom_impact = df.nsmallest(15, 'Recession_Change', keep='first')
    combined = pd.concat([top_impact, bottom_impact]).drop_duplicates()
    combined = combined.sort_values('Recession_Change', ascending=True)

    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in combined['Recession_Change']]
    ax.barh(combined['AREANM'], combined['Recession_Change'], color=colors)

    ax.set_xlabel('Change in GDHI (GBP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Area Name', fontsize=12, fontweight='bold')
    ax.set_title('Recession Impact by Area (2008-2009 Change)', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Income Increased'),
        Patch(facecolor='#e74c3c', label='Income Decreased')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if verbose:
        fig.savefig(OUTPUT_DIR / "week9_recession_impact.png", dpi=300, bbox_inches='tight')
        print("[OK] Saved: week9_recession_impact.png")

    return fig


def create_summary_dashboard(df: pd.DataFrame, verbose: bool = True) -> Figure:
    """Visualization 5: Performance Summary Dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Top 10 by Growth Rate
    ax1 = fig.add_subplot(gs[0, 0])
    top_10_growth = df.nlargest(10, 'Growth_Rate_1997_to_2016')
    ax1.barh(top_10_growth['AREANM'], top_10_growth['Growth_Rate_1997_to_2016'], color='#2ecc71')
    ax1.set_xlabel('Growth Rate (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Top 10 by Growth Rate', fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.tick_params(axis='y', labelsize=8)

    # 2. Top 10 by 2016 Income
    ax2 = fig.add_subplot(gs[0, 1])
    top_10_income = df.nlargest(10, '2016')
    ax2.barh(top_10_income['AREANM'], top_10_income['2016'], color='#3498db')
    ax2.set_xlabel('GDHI 2016 (GBP)', fontsize=10, fontweight='bold')
    ax2.set_title('Top 10 by Income (2016)', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.tick_params(axis='y', labelsize=8)

    # 3. Income Category Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    category_counts = df['Income_Category_2016'].value_counts()
    colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
    ax3.pie(
        category_counts.to_numpy(),
        labels=category_counts.index.astype(str).tolist(),
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
    )
    ax3.set_title('Income Category Distribution', fontsize=11, fontweight='bold')

    # 4. Regional Income Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    regional_avg = df.groupby('Region_Group')['2016'].mean().sort_values()
    if 'Unknown' in regional_avg.index:
        regional_avg = regional_avg.drop('Unknown')
    avg_vals = regional_avg.to_numpy(dtype=float)
    ax4.barh(
        regional_avg.index,
        avg_vals,
        color=cm.get_cmap("viridis")(np.linspace(0, 1, len(regional_avg))),
    )
    ax4.set_xlabel('Average GDHI (GBP)', fontsize=10, fontweight='bold')
    ax4.set_title('Average Income by Region', fontsize=11, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.tick_params(axis='y', labelsize=8)

    # 5. Growth vs Initial Income
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(
        df['1997'].to_numpy(dtype=float),
        df['Growth_Rate_1997_to_2016'].to_numpy(dtype=float),
        alpha=0.6,
        c=df['2016'].to_numpy(dtype=float),
        cmap=cm.get_cmap("RdYlGn"),
        s=50,
    )
    ax5.set_xlabel('Initial Income 1997 (GBP)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Growth Rate (%)', fontsize=10, fontweight='bold')
    ax5.set_title('Growth Rate vs Initial Income', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Recession Impact Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(df['Recession_Change'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
    ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax6.axvline(df['Recession_Change'].mean(), color='blue', linestyle='--',
               linewidth=2, label=f'Mean: GBP {df["Recession_Change"].mean():.0f}')
    ax6.set_xlabel('Income Change (GBP)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Number of Areas', fontsize=10, fontweight='bold')
    ax6.set_title('Recession Impact Distribution (2008-2009)', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.3)

    if verbose:
        fig.savefig(OUTPUT_DIR / "week9_summary_dashboard.png", dpi=300, bbox_inches='tight')
        print("[OK] Saved: week9_summary_dashboard.png")

    return fig


def print_summary_statistics(df: pd.DataFrame, verbose: bool = True) -> None:
    """Print summary statistics for the analysis."""
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total UK Areas: {len(df)}")
    print(f"Average GDHI 2016: GBP {df['2016'].mean():,.2f}")
    print(f"Highest GDHI 2016: GBP {df['2016'].max():,.2f} ({df.loc[df['2016'].idxmax(), 'AREANM']})")
    print(f"Lowest GDHI 2016: GBP {df['2016'].min():,.2f} ({df.loc[df['2016'].idxmin(), 'AREANM']})")
    print(f"Average Growth Rate: {df['Growth_Rate_1997_to_2016'].mean():.2f}%")
    print(f"Regions: {df['Region_Group'].nunique()}")

    print("\nRECESSION IMPACT (2008-2009):")
    print(f"  Areas with Income Increase: {(df['Recession_Change'] > 0).sum()}")
    print(f"  Areas with Income Decrease: {(df['Recession_Change'] < 0).sum()}")
    print(f"  Average Change: GBP {df['Recession_Change'].mean():,.2f}")


def prepare_outputs(path: str | None, verbose: bool = True) -> Tuple[List[Tuple[str, Figure]], str, str]:
    """Prepare all visualizations and summary text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df, actual_path = load_data(path, verbose=True)
        print_summary_statistics(df, verbose=True)

        fig1 = create_top_20_wealthiest_viz(df, verbose=True)
        fig2 = create_growth_rate_viz(df, verbose=True)
        fig3 = create_regional_comparison_viz(df, verbose=True)
        fig4 = create_recession_impact_viz(df, verbose=True)
        fig5 = create_summary_dashboard(df, verbose=True)

    summary = buffer.getvalue()
    if verbose:
        print(summary)

    figures = [
        ("Top 20 Wealthiest Areas", fig1),
        ("Growth Rate Analysis", fig2),
        ("Regional Comparison", fig3),
        ("Recession Impact", fig4),
        ("Performance Summary Dashboard", fig5),
    ]

    return figures, summary, actual_path


def run(config: dict) -> None:
    """Streamlit interface for Week 9."""
    import streamlit as st  # type: ignore

    figures, summary, actual_path = prepare_outputs(config.get("data_path"), verbose=False)
    st.header("Week 9 - Tableau-equivalent visualizations")
    st.caption(f"Data source: {actual_path}")

    labels = [title for title, _ in figures]
    choice = st.selectbox("Select visualization", labels)
    fig_map = dict(figures)
    st.pyplot(fig_map[choice])

    st.markdown("**Summary**")
    st.code(summary, language="text")


def build_widget(config: dict):
    """Qt widget builder for Week 9."""
    figures, summary, _ = prepare_outputs(config.get("data_path"), verbose=False)
    return {"figures": figures, "text": [("Summary", summary)]}


def main() -> None:
    """Main CLI execution for Week 9."""
    prepare_outputs(None, verbose=True)
    print("\nAnalysis complete! Visualizations saved to outputs/")


if __name__ == "__main__":
    main()
