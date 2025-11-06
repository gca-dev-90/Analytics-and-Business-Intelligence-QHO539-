"""Week 3 - Descriptive statistics and visualisation for the GDHI dataset."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from utils.data_loader import default_csv_path, load_csv

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENCY_FORMATTER = mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}")


def load_data(path: str | None) -> tuple[pd.DataFrame, str]:
    actual_path = path or str(default_csv_path())
    df = load_csv(actual_path).copy()
    return df, actual_path


def format_currency(value: float) -> str:
    return f"£{value:,.2f}"


def format_percentage(value: float) -> str:
    return f"{value:.2f}%"


def get_statistics_summary_text(stats: dict[str, float]) -> str:
    lines = [
        "=== OVERALL STATISTICS ===",
        "",
        f"Count:               {stats['Count']:,}",
        f"Mean:                {format_currency(stats['Mean'])}",
        f"Median:              {format_currency(stats['Median'])}",
        f"Mode:                {format_currency(stats['Mode'])}",
        f"Minimum:             {format_currency(stats['Minimum'])}",
        f"Maximum:             {format_currency(stats['Maximum'])}",
        f"Range:               {format_currency(stats['Range'])}",
        f"Standard Deviation:  {format_currency(stats['Standard Deviation'])}",
        f"Variance:            {format_currency(stats['Variance'])}",
        f"Q1 (25th %ile):      {format_currency(stats['Q1 (25th Percentile)'])}",
        f"Median (50th %ile):  {format_currency(stats['Q2 (50th Percentile)'])}",
        f"Q3 (75th %ile):      {format_currency(stats['Q3 (75th Percentile)'])}",
        f"IQR:                 {format_currency(stats['IQR'])}",
    ]
    return "\n".join(lines)


class GDHIAnalyzer:
    """Provide descriptive statistics and visuals for the GDHI dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.year_columns = sorted(
            [col for col in self.df.columns if col not in {"AREANM", "AREACD"}],
            key=lambda x: int(x),
        )

    def calculate_overall_statistics(self) -> dict[str, float]:
        values = self.df[self.year_columns].to_numpy().astype(float).ravel()
        values = values[~np.isnan(values)]

        return {
            "Count": float(len(values)),
            "Mean": float(np.mean(values)),
            "Median": float(np.median(values)),
            "Mode": float(pd.Series(values).mode().iloc[0]),
            "Minimum": float(np.min(values)),
            "Maximum": float(np.max(values)),
            "Range": float(np.max(values) - np.min(values)),
            "Standard Deviation": float(np.std(values, ddof=1)),
            "Variance": float(np.var(values, ddof=1)),
            "Q1 (25th Percentile)": float(np.percentile(values, 25)),
            "Q2 (50th Percentile)": float(np.percentile(values, 50)),
            "Q3 (75th Percentile)": float(np.percentile(values, 75)),
            "IQR": float(np.percentile(values, 75) - np.percentile(values, 25)),
        }

    def calculate_year_by_year_stats(self) -> pd.DataFrame:
        rows = []
        for idx, year in enumerate(self.year_columns):
            avg = self.df[year].mean()
            growth = None
            if idx > 0:
                prev_avg = self.df[self.year_columns[idx - 1]].mean()
                growth = ((avg - prev_avg) / prev_avg) * 100 if prev_avg else None
            rows.append(
                {
                    "Year": int(year),
                    "Average GDHI": avg,
                    "Growth from Previous Year (%)": growth,
                }
            )
        return pd.DataFrame(rows)

    def get_regional_comparison(self, top_n: int = 10) -> pd.DataFrame:
        records = []
        for _, row in self.df.iterrows():
            gdhi_1997 = row["1997"]
            gdhi_2016 = row["2016"]
            avg = row[self.year_columns].mean()
            growth = ((gdhi_2016 - gdhi_1997) / gdhi_1997) * 100
            records.append(
                {
                    "Area Name": row["AREANM"],
                    "1997 GDHI": gdhi_1997,
                    "2016 GDHI": gdhi_2016,
                    "Average GDHI (1997-2016)": avg,
                    "Total Growth (%)": growth,
                }
            )
        df_regional = pd.DataFrame(records).sort_values(
            "Average GDHI (1997-2016)", ascending=False
        )
        return df_regional.head(top_n)

    def calculate_yearly_distribution(self) -> pd.DataFrame:
        return self.df[self.year_columns]

    def create_summary_dashboard(self) -> Figure:
        fig = Figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        stats_df = self.calculate_year_by_year_stats()
        ax1.plot(stats_df["Year"], stats_df["Average GDHI"], marker="o", linewidth=2, color="#4472C4")
        ax1.set_title("UK Average GDHI Trend", fontweight="bold")
        ax1.set_xlabel("Year", fontweight="bold")
        ax1.set_ylabel("Average GDHI (£)", fontweight="bold")
        ax1.yaxis.set_major_formatter(CURRENCY_FORMATTER)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        ax2 = fig.add_subplot(gs[0, 1])
        data_2016 = self.df["2016"].dropna()
        ax2.hist(data_2016, bins=25, color="#4472C4", edgecolor="black", alpha=0.7)
        ax2.axvline(data_2016.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: £{data_2016.mean():,.0f}")
        ax2.set_title("GDHI Distribution (2016)", fontweight="bold")
        ax2.set_xlabel("GDHI (£)", fontweight="bold")
        ax2.set_ylabel("Frequency", fontweight="bold")
        ax2.xaxis.set_major_formatter(CURRENCY_FORMATTER)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")

        ax3 = fig.add_subplot(gs[1, 0])
        regional = self.get_regional_comparison(10)
        ax3.barh(regional["Area Name"], regional["Average GDHI (1997-2016)"], color="#4472C4")
        ax3.set_title("Top 10 Regions by Average GDHI", fontweight="bold")
        ax3.set_xlabel("Average GDHI (£)", fontweight="bold")
        ax3.set_ylabel("Region", fontweight="bold")
        ax3.xaxis.set_major_formatter(CURRENCY_FORMATTER)
        ax3.tick_params(axis="y", labelsize=8)
        ax3.invert_yaxis()

        ax4 = fig.add_subplot(gs[1, 1])
        growth_data = regional.sort_values("Total Growth (%)", ascending=True)
        colors = ["#2E7D32" if x > 0 else "#C62828" for x in growth_data["Total Growth (%)"]]
        ax4.barh(growth_data["Area Name"], growth_data["Total Growth (%)"], color=colors)
        ax4.set_title("Growth Rate (1997-2016)", fontweight="bold")
        ax4.set_xlabel("Growth (%)", fontweight="bold")
        ax4.axvline(0, color="black", linewidth=0.8)
        ax4.tick_params(axis="y", labelsize=8)

        return fig


def prepare_outputs(path: str | None, verbose: bool = True):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df, actual_path = load_data(path)
        analyzer = GDHIAnalyzer(df)

        print(f"Loaded dataset: {actual_path}")
        print()
        print(get_statistics_summary_text(analyzer.calculate_overall_statistics()))
        print()
        print("=== YEAR-BY-YEAR STATISTICS ===")
        print(analyzer.calculate_year_by_year_stats().to_string(index=False))
        print()
        print("=== TOP 10 REGIONS BY AVERAGE GDHI ===")
        print(analyzer.get_regional_comparison(10).to_string(index=False))

        fig = analyzer.create_summary_dashboard()

    summary = buffer.getvalue()
    if verbose:
        print(summary)

    fig.savefig(OUTPUT_DIR / "week3_summary_dashboard.png", dpi=300, bbox_inches="tight")
    figures = [("Inequality Dashboard", fig)]
    return figures, summary, actual_path


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    figures, summary, actual_path = prepare_outputs(config.get("data_path"), verbose=False)
    st.header("Week 3 - Inequality dashboard")
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


if __name__ == "__main__":
    main()
