"""Week 7 - Univariate and multivariate analysis for the GDHI dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from utils.data_loader import default_csv_path, load_csv

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: str | None, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    actual_path = path or str(default_csv_path())
    df = load_csv(actual_path).copy()
    if verbose:
        print("=" * 80)
        print("WEEK 7: UNIVARIATE AND MULTIVARIATE ANALYSIS")
        print("Gross Disposable Household Income (GDHI) Analysis")
        print("=" * 80)
        print("\n1. DATA OVERVIEW")
        print("-" * 80)
        print(f"Dataset shape: {df.shape}")
        print(f"Number of regions: {len(df)}")
        print("Time period: 1997-2016 (20 years)")
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nBasic statistics:")
        print(df.describe())
    return df, actual_path



class StatsContext(TypedDict):
    year_2016: pd.Series
    year_2000: pd.Series
    year_1997: pd.Series
    q1: float
    q3: float
    iqr: float
    skewness_2016: float
    skewness_2000: float
    skewness_1997: float
    kurtosis_2016: float


def _compute_statistics(df: pd.DataFrame, verbose: bool) -> StatsContext:
    year_2016: pd.Series = df["2016"]
    year_2000: pd.Series = df["2000"]
    year_1997: pd.Series = df["1997"]

    stats_ctx: StatsContext = {
        "year_2016": year_2016,
        "year_2000": year_2000,
        "year_1997": year_1997,
        "q1": float(year_2016.quantile(0.25)),
        "q3": float(year_2016.quantile(0.75)),
        "iqr": float(year_2016.quantile(0.75) - year_2016.quantile(0.25)),
        "skewness_2016": float(stats.skew(year_2016.to_numpy())),
        "skewness_2000": float(stats.skew(year_2000.to_numpy())),
        "skewness_1997": float(stats.skew(year_1997.to_numpy())),
        "kurtosis_2016": float(stats.kurtosis(year_2016.to_numpy())),
    }

    if verbose:
        print("\n\n" + "=" * 80)
        print("2. UNIVARIATE ANALYSIS")
        print("=" * 80)
        print("\n2.1 Descriptive Statistics for GDHI in 2016")
        print("-" * 80)
        print(f"Mean: GBP {year_2016.mean():.2f}")
        print(f"Median: GBP {year_2016.median():.2f}")
        print(f"Standard Deviation: GBP {year_2016.std():.2f}")
        print(f"Minimum: GBP {year_2016.min():.2f}")
        print(f"Maximum: GBP {year_2016.max():.2f}")
        print(f"Range: GBP {year_2016.max() - year_2016.min():.2f}")
        print(f"\nQ1 (25th percentile): GBP {stats_ctx['q1']:.2f}")
        print(f"Q3 (75th percentile): GBP {stats_ctx['q3']:.2f}")
        print(f"IQR: GBP {stats_ctx['iqr']:.2f}")

        print("\n\n2.2 Skewness Analysis")
        print("-" * 80)
        print(f"Skewness in 1997: {stats_ctx['skewness_1997']:.4f}")
        print(f"Skewness in 2000: {stats_ctx['skewness_2000']:.4f}")
        print(f"Skewness in 2016: {stats_ctx['skewness_2016']:.4f}")

        skew = stats_ctx["skewness_2016"]
        if abs(skew) < 0.5:
            interpretation = "approximately symmetric"
        elif skew > 0:
            interpretation = "positively skewed (right tail)"
        else:
            interpretation = "negatively skewed (left tail)"
        print(f"\nInterpretation: Distribution is {interpretation}")

        kurt = stats_ctx["kurtosis_2016"]
        if kurt > 0:
            kurt_txt = "heavier tails than normal (leptokurtic)"
        elif kurt < 0:
            kurt_txt = "lighter tails than normal (platykurtic)"
        else:
            kurt_txt = "close to normal (mesokurtic)"
        print(f"Kurtosis in 2016: {kurt:.4f} -> {kurt_txt}")

    return stats_ctx


def _figure_univariate(df: pd.DataFrame, stats_ctx: StatsContext) -> Figure:
    year_2016: pd.Series = stats_ctx["year_2016"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(year_2016, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(year_2016.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: GBP {year_2016.mean():.0f}")
    axes[0, 0].axvline(year_2016.median(), color="green", linestyle="--", linewidth=2, label=f"Median: GBP {year_2016.median():.0f}")
    axes[0, 0].set_title("Histogram of GDHI Distribution", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    bp = axes[0, 1].boxplot(year_2016, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    axes[0, 1].set_title("Box Plot - Identifying Outliers", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(1.15, year_2016.median(), f"Median: GBP {year_2016.median():.0f}", va="center", fontsize=9)
    axes[0, 1].text(1.15, stats_ctx["q1"], f"Q1: GBP {stats_ctx['q1']:.0f}", va="center", fontsize=9)
    axes[0, 1].text(1.15, stats_ctx["q3"], f"Q3: GBP {stats_ctx['q3']:.0f}", va="center", fontsize=9)

    axes[1, 0].hist(year_2016, bins=30, density=True, alpha=0.5, color="skyblue", edgecolor="black")
    year_2016.plot(kind="density", ax=axes[1, 0], color="darkblue", linewidth=2)
    axes[1, 0].set_title("Probability Density Function (KDE)", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    stats.probplot(year_2016.to_numpy(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot - Normal Distribution Test", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _figure_time_series(df: pd.DataFrame, verbose: bool) -> Tuple[Figure, float, float]:
    years = [str(year) for year in range(1997, 2017)]
    mean_gdhi_by_year = df[years].mean().astype(float)
    years_numeric = np.arange(len(years))
    growth_rate = ((mean_gdhi_by_year["2016"] - mean_gdhi_by_year["1997"]) / mean_gdhi_by_year["1997"]) * 100
    annual_growth = growth_rate / 19

    if verbose:
        print("\n\n2.4 Time Series Analysis")
        print("-" * 80)
        print(f"Overall growth in average GDHI (1997-2016): {growth_rate:.2f}%")
        print(f"Average annual growth rate: {annual_growth:.2f}%")

    fig = plt.figure(figsize=(14, 6))
    values = mean_gdhi_by_year.to_numpy(dtype=float)
    plt.plot(years_numeric, values, marker="o", linewidth=2, markersize=6, color="darkblue")
    plt.fill_between(years_numeric, values, alpha=0.3, color="skyblue")
    plt.xlabel("Year")
    plt.ylabel("Average GDHI per head (GBP)")
    plt.title("Average GDHI Trend Across All UK Regions (1997-2016)", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(years_numeric, years, rotation=45)
    start_val = float(mean_gdhi_by_year["1997"])
    end_val = float(mean_gdhi_by_year["2016"])
    plt.annotate(
        f"Start: GBP {start_val:.0f}",
        xy=(0, start_val),
        xytext=(2, start_val + 1000.0),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )
    plt.annotate(
        f"End: GBP {end_val:.0f}",
        xy=(len(years) - 1, end_val),
        xytext=(len(years) - 5, end_val + 1000.0),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=10,
        color="green",
    )
    fig.tight_layout()
    return fig, float(growth_rate), float(annual_growth)


def _figure_correlation(df: pd.DataFrame) -> Tuple[Figure, Dict[str, float]]:
    selected_years = ["1997", "2000", "2005", "2010", "2016"]
    corr_matrix = df[selected_years].corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".3f", square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix: GDHI Across Selected Years", fontweight="bold")
    plt.tight_layout()

    corr_values = {
        "1997_2016": float(corr_matrix.loc["1997", "2016"]),
        "2000_2010": float(corr_matrix.loc["2000", "2010"]),
        "2010_2016": float(corr_matrix.loc["2010", "2016"]),
    }
    return fig, corr_values


def _figure_scatter(df: pd.DataFrame) -> Tuple[Figure, Dict[str, float]]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plots = [
        ("1997", "2016", "1997 vs 2016"),
        ("2000", "2010", "2000 vs 2010"),
        ("2010", "2016", "2010 vs 2016"),
    ]
    corr_values: Dict[str, float] = {}

    for ax, (x_col, y_col, title) in zip(axes, plots):
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, color="steelblue", edgecolors="black", linewidth=0.5)
        ax.plot([df[x_col].min(), df[x_col].max()], [df[x_col].min(), df[x_col].max()], "r--", linewidth=2)
        ax.set_xlabel(f"GDHI {x_col} (GBP)")
        ax.set_ylabel(f"GDHI {y_col} (GBP)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)
        corr = float(df[x_col].corr(df[y_col]))
        corr_values[f"{x_col}_{y_col}"] = corr
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    return fig, corr_values


def _simple_linear_regression(df: pd.DataFrame) -> Tuple[LinearRegression, np.ndarray, np.ndarray, float, float, float, Figure]:
    X = df["2010"].to_numpy(dtype=float).reshape(-1, 1)
    y = df["2016"].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    fig = plt.figure(figsize=(12, 7))
    plt.scatter(X, y, alpha=0.6, s=60, color="steelblue", edgecolors="black", linewidth=0.5, label="Actual data")
    plt.plot(X, y_pred, color="red", linewidth=2.5, label="Regression line")
    residuals = y - y_pred
    std_resids = float(np.std(residuals))
    plt.fill_between(
        X.flatten(),
        y_pred - 1.96 * std_resids,
        y_pred + 1.96 * std_resids,
        alpha=0.2,
        color="red",
        label="95% Confidence Interval",
    )
    plt.xlabel("GDHI 2010 (GBP)")
    plt.ylabel("GDHI 2016 (GBP)")
    plt.title("Linear Regression: Predicting 2016 GDHI from 2010 GDHI", fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.text(
        0.05,
        0.95,
        f"y = {model.intercept_:.2f} + {model.coef_[0]:.4f}x\nR^2 = {r2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    fig.tight_layout()
    return model, X.flatten(), y, float(r2), float(mse), float(rmse), fig


def _residual_figure(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, Figure]:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, color="purple", edgecolors="black", linewidth=0.5)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicted GDHI 2016 (GBP)")
    axes[0].set_ylabel("Residuals (GBP)")
    axes[0].set_title("Residual Plot", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=25, color="lightgreen", edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero line")
    axes[1].set_title("Distribution of Residuals", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return residuals, fig


def _multiple_linear_regression(df: pd.DataFrame) -> Tuple[LinearRegression, float, float, float, Figure]:
    X_multi = df[["2000", "2005", "2010"]].to_numpy(dtype=float)
    y_multi = df["2016"].to_numpy(dtype=float)
    model = LinearRegression().fit(X_multi, y_multi)
    y_pred = model.predict(X_multi)
    r2 = r2_score(y_multi, y_pred)
    mse = mean_squared_error(y_multi, y_pred)
    rmse = np.sqrt(mse)

    fig = plt.figure(figsize=(12, 7))
    plt.scatter(y_multi, y_pred, alpha=0.6, s=60, color="darkgreen", edgecolors="black", linewidth=0.5)
    plt.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], "r--", linewidth=2.5)
    plt.xlabel("Actual GDHI 2016 (GBP)")
    plt.ylabel("Predicted GDHI 2016 (GBP)")
    plt.title("Multiple Linear Regression: Actual vs Predicted Values", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.text(
        0.05,
        0.95,
        f"R^2 = {r2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )
    fig.tight_layout()
    return model, float(r2), float(mse), float(rmse), fig


def _regional_comparison(df: pd.DataFrame) -> Tuple[Figure, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["growth_1997_2016"] = ((df["2016"] - df["1997"]) / df["1997"]) * 100
    top_regions = df.nlargest(5, "growth_1997_2016")[["AREANM", "growth_1997_2016", "1997", "2016"]]
    bottom_regions = df.nsmallest(5, "growth_1997_2016")[["AREANM", "growth_1997_2016", "1997", "2016"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(top_regions["AREANM"].str[:20], top_regions["growth_1997_2016"], color="green", alpha=0.7, edgecolor="black")
    axes[0].set_title("Top 5 Regions by GDHI Growth (1997-2016)", fontweight="bold")
    axes[0].set_xlabel("Growth Rate (%)")
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].barh(bottom_regions["AREANM"].str[:20], bottom_regions["growth_1997_2016"], color="red", alpha=0.7, edgecolor="black")
    axes[1].set_title("Bottom 5 Regions by GDHI Growth (1997-2016)", fontweight="bold")
    axes[1].set_xlabel("Growth Rate (%)")
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return fig, top_regions, bottom_regions


def _build_summary(
    stats_ctx: StatsContext,
    growth_rate: float,
    annual_growth: float,
    corr_values: Dict[str, float],
    simple_metrics: Dict[str, float],
    multi_r2: float,
    top_regions: pd.DataFrame,
    bottom_regions: pd.DataFrame,
    verbose: bool,
) -> str:
    lines: List[str] = []
    lines.append("\n" + "=" * 80)
    lines.append("6. SUMMARY AND KEY FINDINGS")
    lines.append("=" * 80)
    lines.append("\nUNIVARIATE ANALYSIS FINDINGS:")
    lines.append("-" * 80)
    lines.append(f"- Average GDHI in 2016: GBP {stats_ctx['year_2016'].mean():.2f}")
    skew = stats_ctx["skewness_2016"]
    skew_txt = "approximately symmetric" if abs(skew) < 0.5 else "positively skewed" if skew > 0 else "negatively skewed"
    lines.append(f"- Distribution skewness: {skew:.4f} -> {skew_txt}")
    lines.append(f"- Income inequality range: GBP {stats_ctx['year_2016'].max() - stats_ctx['year_2016'].min():.2f}")

    lines.append("\nTIME SERIES FINDINGS:")
    lines.append("-" * 80)
    lines.append(f"- Overall growth (1997-2016): {growth_rate:.2f}%")
    lines.append(f"- Average annual growth: {annual_growth:.2f}%")
    lines.append("- Trend: steady increase with a slowdown around 2008-2010 (financial crisis)")

    lines.append("\nMULTIVARIATE ANALYSIS FINDINGS:")
    lines.append("-" * 80)
    lines.append(f"- Correlation 1997->2016: {corr_values['1997_2016']:.4f}")
    lines.append(f"- Simple Linear Regression R^2: {simple_metrics['r2']:.4f} ({simple_metrics['r2']*100:.2f}% variance explained)")
    lines.append(f"- Multiple Linear Regression R^2: {multi_r2:.4f} ({multi_r2*100:.2f}% variance explained)")

    lines.append("\nREGRESSION INSIGHTS:")
    lines.append("-" * 80)
    lines.append(f"- GBP 1 increase in 2010 GDHI -> GBP {simple_metrics['coef']:.2f} rise in 2016 GDHI")
    lines.append(f"- Multiple regression improves prediction by {(multi_r2 - simple_metrics['r2'])*100:.2f}%")
    lines.append(f"- RMSE: GBP {simple_metrics['rmse']:.2f}")

    lines.append("\nREGIONAL INSIGHTS:")
    lines.append("-" * 80)
    lines.append(
        f"- Highest growth: {top_regions.iloc[0]['AREANM']} ({top_regions.iloc[0]['growth_1997_2016']:.2f}%)"
    )
    lines.append(
        f"- Lowest growth: {bottom_regions.iloc[0]['AREANM']} ({bottom_regions.iloc[0]['growth_1997_2016']:.2f}%)"
    )
    lines.append("- Structural regional differences persist over the 20-year period.")

    lines.append("\n" + "=" * 80)
    lines.append("Analysis complete! Visualizations saved to outputs/")
    lines.append("=" * 80)
    lines.append("\nGenerated files:")
    for filename in [
        "week7_univariate_analysis.png",
        "week7_time_series.png",
        "week7_correlation_matrix.png",
        "week7_scatter_plots.png",
        "week7_linear_regression.png",
        "week7_residual_analysis.png",
        "week7_multiple_regression.png",
        "week7_regional_comparison.png",
    ]:
        lines.append(f"  - {filename}")
    lines.append("=" * 80)

    summary = "\n".join(lines)
    if verbose:
        print(summary)
    return summary


def _prepare_outputs(df: pd.DataFrame, verbose: bool = False) -> Tuple[List[Tuple[str, Figure]], str]:
    stats_ctx = _compute_statistics(df, verbose)
    fig_uni = _figure_univariate(df, stats_ctx)
    fig_ts, growth_rate, annual_growth = _figure_time_series(df, verbose)
    fig_corr, corr_values = _figure_correlation(df)
    fig_scatter, scatter_corrs = _figure_scatter(df)
    model_simple, X_flat, y_true, r2_simple, mse_simple, rmse_simple, fig_lr = _simple_linear_regression(df)
    y_pred = model_simple.predict(X_flat.reshape(-1, 1))
    _, fig_residual = _residual_figure(y_true, y_pred)
    _, r2_multi, mse_multi, rmse_multi, fig_multi = _multiple_linear_regression(df)
    fig_regions, top_regions, bottom_regions = _regional_comparison(df)

    simple_metrics = {"r2": r2_simple, "rmse": rmse_simple, "coef": model_simple.coef_[0]}
    summary_text = _build_summary(
        stats_ctx,
        growth_rate,
        annual_growth,
        {"1997_2016": corr_values["1997_2016"]},
        simple_metrics,
        r2_multi,
        top_regions,
        bottom_regions,
        verbose,
    )

    figures = [
        ("Univariate Analysis", fig_uni),
        ("Time Series Trend", fig_ts),
        ("Correlation Heatmap", fig_corr),
        ("Scatter Plots", fig_scatter),
        ("Linear Regression", fig_lr),
        ("Residual Analysis", fig_residual),
        ("Multiple Regression", fig_multi),
        ("Regional Comparison", fig_regions),
    ]

    if verbose:
        save_map = {
            "week7_univariate_analysis.png": fig_uni,
            "week7_time_series.png": fig_ts,
            "week7_correlation_matrix.png": fig_corr,
            "week7_scatter_plots.png": fig_scatter,
            "week7_linear_regression.png": fig_lr,
            "week7_residual_analysis.png": fig_residual,
            "week7_multiple_regression.png": fig_multi,
            "week7_regional_comparison.png": fig_regions,
        }
        for filename, figure in save_map.items():
            figure.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")

    return figures, summary_text


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    df, actual_path = load_data(config.get("data_path"), verbose=False)
    figures, summary_text = _prepare_outputs(df, verbose=False)

    st.header("Week 7 - Univariate & Multivariate analysis")
    st.caption(f"Data source: {actual_path}")
    labels = [title for title, _ in figures]
    choice = st.selectbox("Select visualization", labels)
    fig_map = dict(figures)
    st.pyplot(fig_map[choice])
    st.markdown("**Summary**")
    st.code(summary_text, language="text")


def build_widget(config: dict):
    df, _ = load_data(config.get("data_path"), verbose=False)
    figures, summary_text = _prepare_outputs(df, verbose=False)
    return {"figures": figures, "text": [("Summary", summary_text)]}


def main() -> None:
    df, _ = load_data(None, verbose=True)
    _prepare_outputs(df, verbose=True)


if __name__ == "__main__":
    main()
