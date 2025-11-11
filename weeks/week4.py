"""Week 4 - Data Quality, Preparation & Exploration for GDHI dataset."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns

from utils.data_loader import default_csv_path, load_csv

YEAR_COLUMNS = [str(year) for year in range(1997, 2017)]
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive data quality checks.

    Returns:
        Dictionary containing quality check results
    """
    results = {}

    # 1. Check for missing values
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    results['missing_values'] = pd.DataFrame({
        'Count': missing_counts,
        'Percentage': missing_pct
    }).query('Count > 0')

    # 2. Check for duplicates
    duplicate_rows = df.duplicated().sum()
    results['duplicate_rows'] = duplicate_rows

    # Check for duplicate areas
    if 'AREANM' in df.columns:
        duplicate_areas = df['AREANM'].duplicated().sum()
        results['duplicate_areas'] = duplicate_areas

    # 3. Validate data types
    results['data_types'] = df.dtypes

    # 4. Check for outliers using Z-scores
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_dict = {}

    for col in numeric_cols:
        if col not in ['AREACD']:  # Skip ID columns
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_count = (z_scores > 3).sum()
            if outlier_count > 0:
                outliers_dict[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100
                }

    results['outliers'] = outliers_dict

    return results


def explore_data(df: pd.DataFrame) -> dict:
    """
    Explore data structure and basic statistics.

    Returns:
        Dictionary containing exploration results
    """
    results = {}

    # 1. Data structure
    results['shape'] = df.shape
    results['columns'] = list(df.columns)

    # 2. First and last rows
    results['first_rows'] = df.head()
    results['last_rows'] = df.tail()

    # 3. Summary statistics
    results['summary_stats'] = df.describe()

    # 4. Identify data types (numeric vs categorical)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    results['numeric_columns'] = numeric_cols
    results['categorical_columns'] = categorical_cols
    results['numeric_count'] = len(numeric_cols)
    results['categorical_count'] = len(categorical_cols)

    return results


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data with calculated columns and classifications.

    Returns:
        DataFrame with additional prepared columns
    """
    df_prep = df.copy()

    # 1. Create calculated columns

    # Growth rates (absolute and percentage)
    df_prep['Growth_GBP'] = df_prep['2016'] - df_prep['1997']
    df_prep['Growth_Pct'] = ((df_prep['2016'] - df_prep['1997']) / df_prep['1997']) * 100

    # Average GDHI across all years
    df_prep['Avg_GDHI'] = df_prep[YEAR_COLUMNS].mean(axis=1)

    # Compound Annual Growth Rate (CAGR)
    years_span = len(YEAR_COLUMNS) - 1
    ratio = (df_prep['2016'] / df_prep['1997']).replace([np.inf, -np.inf], np.nan)
    df_prep['CAGR_Pct'] = (ratio ** (1 / years_span) - 1) * 100

    # Standard deviation (volatility measure)
    df_prep['Std_Dev'] = df_prep[YEAR_COLUMNS].std(axis=1)

    # Coefficient of variation (normalized volatility)
    df_prep['Coeff_Variation'] = (df_prep['Std_Dev'] / df_prep['Avg_GDHI']) * 100

    # 2. Classify regions by growth performance
    growth_median = df_prep['Growth_Pct'].median()
    growth_q1 = df_prep['Growth_Pct'].quantile(0.25)
    growth_q3 = df_prep['Growth_Pct'].quantile(0.75)

    def classify_growth(growth_pct):
        if pd.isna(growth_pct):
            return 'Unknown'
        elif growth_pct >= growth_q3:
            return 'High Growth'
        elif growth_pct >= growth_median:
            return 'Medium-High Growth'
        elif growth_pct >= growth_q1:
            return 'Medium-Low Growth'
        else:
            return 'Low Growth'

    df_prep['Growth_Category'] = df_prep['Growth_Pct'].apply(classify_growth)

    # 3. Classify regions by income level (based on 2016 values)
    income_median = df_prep['2016'].median()
    income_q1 = df_prep['2016'].quantile(0.25)
    income_q3 = df_prep['2016'].quantile(0.75)

    def classify_income(income):
        if pd.isna(income):
            return 'Unknown'
        elif income >= income_q3:
            return 'High Income'
        elif income >= income_median:
            return 'Medium-High Income'
        elif income >= income_q1:
            return 'Medium-Low Income'
        else:
            return 'Low Income'

    df_prep['Income_Category'] = df_prep['2016'].apply(classify_income)

    # 4. Handle data inconsistencies

    # Replace any infinities with NaN
    df_prep.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ensure no negative values in GDHI columns (they should be positive)
    for col in YEAR_COLUMNS:
        if (df_prep[col] < 0).any():
            print(f"Warning: Negative values found in {col}")

    return df_prep


def create_visualizations(df: pd.DataFrame, df_prep: pd.DataFrame) -> Figure:
    """
    Create comprehensive visualizations for data quality and exploration.

    Args:
        df: Original DataFrame
        df_prep: Prepared DataFrame with calculated columns
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.1, 1.6], hspace=0.65, wspace=0.35)

    # 1. Box plots showing outliers
    ax1 = fig.add_subplot(gs[0, :2])
    sample_years = ['1997', '2004', '2010', '2016']
    df[sample_years].boxplot(ax=ax1, patch_artist=True)
    ax1.set_title('Box Plots: GDHI Distribution by Year (Outlier Detection)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDHI per head (£)')
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of 2016 GDHI distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['2016'].dropna(), bins=30, color='#4472C4', alpha=0.7, edgecolor='black')
    ax2.set_title('Distribution: GDHI 2016', fontweight='bold', fontsize=11)
    ax2.set_xlabel('GDHI per head (£)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    # 3. Histogram of Growth Percentage
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(df_prep['Growth_Pct'].dropna(), bins=30, color='#70AD47', alpha=0.7, edgecolor='black')
    ax3.set_title('Distribution: Growth % (1997-2016)', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Growth %')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)

    # 4. Histogram of CAGR
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df_prep['CAGR_Pct'].dropna(), bins=30, color='#FFC000', alpha=0.7, edgecolor='black')
    ax4.set_title('Distribution: CAGR %', fontweight='bold', fontsize=11)
    ax4.set_xlabel('CAGR %')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)

    # 5. Box plot for growth by category
    ax5 = fig.add_subplot(gs[1, 2])
    growth_categories = df_prep.groupby('Growth_Category')['Growth_Pct'].apply(list)
    ax5.boxplot(list(growth_categories.values), tick_labels=list(growth_categories.index))
    ax5.set_title('Growth Distribution by Category', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Growth Category')
    ax5.set_ylabel('Growth %')
    ax5.tick_params(axis='x', rotation=30)
    ax5.grid(True, alpha=0.3)

    # 6. Z-score visualization for outliers
    ax6 = fig.add_subplot(gs[2, 0])
    z_scores = np.abs((df['2016'] - df['2016'].mean()) / df['2016'].std())
    ax6.scatter(range(len(z_scores)), z_scores, alpha=0.6, c=z_scores, cmap='YlOrRd')
    ax6.axhline(y=3, color='red', linestyle='--', linewidth=2, label='Z-score = 3 (outlier threshold)')
    ax6.set_title('Z-Score Analysis: GDHI 2016', fontweight='bold', fontsize=11)
    ax6.set_xlabel('Region Index')
    ax6.set_ylabel('|Z-Score|')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Correlation heatmap
    ax7 = fig.add_subplot(gs[2, 1])
    # Select sample years for correlation analysis
    correlation_years = ['1997', '2000', '2004', '2008', '2012', '2016']
    available_years = [year for year in correlation_years if year in df.columns]

    if len(available_years) >= 2:
        correlation_matrix = df[available_years].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, ax=ax7, cbar_kws={'label': 'Correlation'},
                   square=True, linewidths=0.5)
        ax7.set_title('Correlation Heatmap: GDHI by Year', fontweight='bold', fontsize=11)
        ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right')
        ax7.set_yticklabels(ax7.get_yticklabels(), rotation=0)
    else:
        ax7.text(0.5, 0.5, 'Insufficient data for correlation',
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax7.set_title('Correlation Heatmap: GDHI by Year', fontweight='bold', fontsize=11)
        ax7.axis('off')

    # 8. Summary statistics table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('tight')
    ax8.axis('off')

    summary_data = [
        ['Total Regions', f"{len(df)}"],
        ['Columns', f"{len(df.columns)}"],
        ['Missing Values', f"{df.isnull().sum().sum()}"],
        ['Duplicates', f"{df.duplicated().sum()}"],
        ['Avg Growth %', f"{df_prep['Growth_Pct'].mean():.2f}%"],
        ['Median GDHI 2016', f"£{df['2016'].median():.2f}"],
    ]

    table = ax8.table(cellText=summary_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style the table
    for i in range(len(summary_data)):
        table[(i, 0)].set_facecolor('#E7E6E6')
        table[(i, 0)].set_text_props(weight='bold')

    ax8.set_title('Data Quality Summary', fontweight='bold', fontsize=11)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05)

    return fig


def print_summary(df: pd.DataFrame, df_prep: pd.DataFrame,
                 quality_results: dict, explore_results: dict) -> None:
    """Print comprehensive summary of data quality and exploration."""

    print("=" * 80)
    print("WEEK 4: DATA QUALITY, PREPARATION & EXPLORATION")
    print("=" * 80)

    # Data Structure
    print("\n1. DATA STRUCTURE")
    print("-" * 80)
    print(f"Rows: {explore_results['shape'][0]}")
    print(f"Columns: {explore_results['shape'][1]}")
    print(f"Numeric columns: {explore_results['numeric_count']}")
    print(f"Categorical columns: {explore_results['categorical_count']}")

    # Data Quality Checks
    print("\n2. DATA QUALITY CHECKS")
    print("-" * 80)

    # Missing values
    print("\nMissing Values:")
    if len(quality_results['missing_values']) > 0:
        print(quality_results['missing_values'].to_string())
    else:
        print("  No missing values detected [OK]")

    # Duplicates
    print(f"\nDuplicate Rows: {quality_results['duplicate_rows']}")
    if 'duplicate_areas' in quality_results:
        print(f"Duplicate Areas: {quality_results['duplicate_areas']}")

    # Outliers
    print("\nOutliers (Z-score > 3):")
    if quality_results['outliers']:
        for col, info in quality_results['outliers'].items():
            print(f"  {col}: {info['count']} outliers ({info['percentage']:.2f}%)")
    else:
        print("  No significant outliers detected [OK]")

    # Data Exploration
    print("\n3. DATA EXPLORATION")
    print("-" * 80)

    print("\nFirst 3 Rows:")
    print(explore_results['first_rows'].head(3).to_string())

    print("\nSummary Statistics (Selected Columns):")
    stats_cols = ['1997', '2016'] if '2016' in df.columns else []
    if stats_cols:
        print(explore_results['summary_stats'][stats_cols].to_string())

    # Data Preparation Results
    print("\n4. DATA PREPARATION - CALCULATED COLUMNS")
    print("-" * 80)

    print("\nNew Calculated Columns:")
    calc_columns = ['Growth_GBP', 'Growth_Pct', 'Avg_GDHI', 'CAGR_Pct',
                   'Std_Dev', 'Coeff_Variation']
    existing_calc = [col for col in calc_columns if col in df_prep.columns]
    for col in existing_calc:
        print(f"  - {col}")

    print("\nGrowth Statistics:")
    print(f"  Average Growth (1997-2016): {df_prep['Growth_Pct'].mean():.2f}%")
    print(f"  Median Growth: {df_prep['Growth_Pct'].median():.2f}%")
    print(f"  Max Growth: {df_prep['Growth_Pct'].max():.2f}%")
    print(f"  Min Growth: {df_prep['Growth_Pct'].min():.2f}%")

    print("\nRegion Classification (Growth Category):")
    growth_dist = df_prep['Growth_Category'].value_counts()
    for category, count in growth_dist.items():
        print(f"  {category}: {count} regions ({count/len(df_prep)*100:.1f}%)")

    print("\nRegion Classification (Income Category):")
    income_dist = df_prep['Income_Category'].value_counts()
    for category, count in income_dist.items():
        print(f"  {category}: {count} regions ({count/len(df_prep)*100:.1f}%)")

    # Top and Bottom Performers
    print("\n5. TOP & BOTTOM PERFORMERS")
    print("-" * 80)

    print("\nTop 5 Regions by Growth:")
    top_growth = df_prep.nlargest(5, 'Growth_Pct')[['AREANM', 'Growth_Pct', 'CAGR_Pct', '2016']]
    for idx, row in top_growth.iterrows():
        print(f"  {row['AREANM']}: {row['Growth_Pct']:.2f}% growth, "
              f"CAGR {row['CAGR_Pct']:.2f}%, 2016 GDHI £{row['2016']:.2f}")

    print("\nBottom 5 Regions by Growth:")
    bottom_growth = df_prep.nsmallest(5, 'Growth_Pct')[['AREANM', 'Growth_Pct', 'CAGR_Pct', '2016']]
    for idx, row in bottom_growth.iterrows():
        print(f"  {row['AREANM']}: {row['Growth_Pct']:.2f}% growth, "
              f"CAGR {row['CAGR_Pct']:.2f}%, 2016 GDHI £{row['2016']:.2f}")

    print("\n" + "=" * 80)


def prepare_outputs(path: str | None, verbose: bool = True) -> Tuple[List[Tuple[str, Figure]], str, str]:
    """
    Main function to prepare all outputs for Week 4.

    Returns:
        Tuple of (figures, summary_text, data_path)
    """
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        # Load data
        actual_path = path or str(default_csv_path())
        df = load_csv(actual_path).copy()

        # Perform data quality checks
        quality_results = check_data_quality(df)

        # Explore data
        explore_results = explore_data(df)

        # Prepare data with calculated columns
        df_prep = prepare_data(df)

        # Print comprehensive summary
        print_summary(df, df_prep, quality_results, explore_results)

        # Create visualizations
        fig = create_visualizations(df, df_prep)

    summary = buffer.getvalue()

    if verbose:
        print(summary)
        fig.savefig(OUTPUT_DIR / "week4_data_quality_exploration.png",
                   dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {OUTPUT_DIR / 'week4_data_quality_exploration.png'}")

    figures = [("Data Quality & Exploration Dashboard", fig)]
    return figures, summary, actual_path


def run(config: dict) -> None:
    """Run Week 4 analysis (Streamlit interface)."""
    import streamlit as st  # type: ignore

    figures, summary, actual_path = prepare_outputs(config.get("data_path"), verbose=False)

    st.header("Week 4 - Data Quality, Preparation & Exploration")
    st.caption(f"Data source: {actual_path}")

    if len(figures) == 1:
        label, figure = figures[0]
        st.pyplot(figure)
    else:
        labels = [title for title, _ in figures]
        choice = st.selectbox("Select visualisation", labels)
        fig_map = dict(figures)
        st.pyplot(fig_map[choice])

    st.markdown("**Analysis Summary**")
    st.code(summary, language="text")


def build_widget(config: dict):
    """Build widget for PyQt interface."""
    figures, summary, _ = prepare_outputs(config.get("data_path"), verbose=False)
    return {"figures": figures, "text": [("Summary", summary)]}


def main() -> None:
    """Main entry point for standalone execution."""
    prepare_outputs(None, verbose=True)


if __name__ == "__main__":
    main()
