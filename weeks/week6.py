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
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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
        print("=" * 70)
        print("WEEK 6: MACHINE LEARNING WITH GDHI DATA")
        print("=" * 70)
        print("\n1. DATA OVERVIEW")
        print("-" * 70)
        print(f"Dataset shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print(f"\nMissing values: {df.isnull().sum().sum()} total missing values")
    return df, actual_path


def engineer_features(
    df: pd.DataFrame, verbose: bool = True
) -> tuple[pd.DataFrame, list[str], Figure]:
    years = [str(year) for year in range(1997, 2017)]
    df_numeric = df[years].apply(pd.to_numeric, errors="coerce")

    derived = df.copy()
    derived["avg_gdhi"] = df_numeric.mean(axis=1)
    derived["total_growth"] = df_numeric["2016"] - df_numeric["1997"]
    derived["growth_rate"] = ((df_numeric["2016"] - df_numeric["1997"]) / df_numeric["1997"]) * 100

    recent_years = [str(year) for year in range(2010, 2017)]
    derived["recent_avg"] = df_numeric[recent_years].mean(axis=1)

    derived["volatility"] = df_numeric.std(axis=1)
    early_period = [str(year) for year in range(1997, 2007)]
    late_period = [str(year) for year in range(2007, 2017)]
    derived["early_avg"] = df_numeric[early_period].mean(axis=1)
    derived["late_avg"] = df_numeric[late_period].mean(axis=1)
    derived["period_change"] = derived["late_avg"] - derived["early_avg"]

    features_created = [
        "avg_gdhi",
        "total_growth",
        "growth_rate",
        "recent_avg",
        "volatility",
        "early_avg",
        "late_avg",
        "period_change",
    ]

    corr = derived[features_created].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True, linewidths=1, ax=ax_corr)
    ax_corr.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig_corr.tight_layout()
    fig_corr.savefig(OUTPUT_DIR / "week6_feature_correlation.png", dpi=300, bbox_inches="tight")

    if verbose:
        print("\n" + "=" * 70)
        print("PART 1: FEATURE ENGINEERING & SELECTION")
        print("=" * 70)
        print("\nCreated Features:")
        print(features_created)
        print("\nFeature Statistics:")
        print(derived[features_created].describe())
        print("\n" + "-" * 70)
        print("FEATURE CORRELATION ANALYSIS")
        print("-" * 70)
        print(corr)
        print("[OK] Saved: week6_feature_correlation.png")

    selected_features = ["avg_gdhi", "growth_rate", "volatility", "period_change"]
    if verbose:
        print(f"\nSelected features for downstream models: {selected_features}")
    return derived, selected_features, fig_corr


def univariate_analysis(df: pd.DataFrame, features: list[str], verbose: bool = True) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, feature in enumerate(features):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        series = df[feature].dropna()
        ax.hist(series, bins=30, alpha=0.7, edgecolor="black", density=True)
        series.plot(kind="kde", ax=ax, color="red", linewidth=2)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {feature}", fontsize=12, fontweight="bold")
        ax.axvline(series.mean(), color="blue", linestyle="--", linewidth=2, label=f"Mean: {series.mean():.2f}")
        ax.axvline(series.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {series.median():.2f}")
        ax.legend()

    fig.tight_layout()
    if verbose:
        fig.savefig(OUTPUT_DIR / "week6_univariate_analysis.png", dpi=300, bbox_inches="tight")
        print("[OK] Saved: week6_univariate_analysis.png")
        print("\nUnivariate summary statistics:")
        print(df[features].describe())
    return fig


def multivariate_analysis(df: pd.DataFrame, features: list[str], verbose: bool = True) -> List[Figure]:
    figs: List[Figure] = []
    if verbose:
        print("\n" + "=" * 70)
        print("PART 3: MULTIVARIATE ANALYSIS")
        print("=" * 70)

    scatter_matrix(df[features], figsize=(12, 12), diagonal="kde", alpha=0.7)
    fig_sm = plt.gcf()
    fig_sm.tight_layout()
    fig_sm.savefig(OUTPUT_DIR / "week6_scatter_matrix.png", dpi=300, bbox_inches="tight")
    figs.append(fig_sm)
    if verbose:
        print("[OK] Saved: week6_scatter_matrix.png")

    categories = pd.qcut(df["avg_gdhi"], q=3, labels=["Low", "Medium", "High"])
    fig_cat, ax_cat = plt.subplots(figsize=(10, 6))
    for label, color in zip(["Low", "Medium", "High"], ["#1b9e77", "#d95f02", "#7570b3"]):
        mask = categories == label
        ax_cat.scatter(df.loc[mask, "avg_gdhi"], df.loc[mask, "growth_rate"], label=label, alpha=0.7, color=color)
    ax_cat.set_xlabel("Average GDHI (GBP)")
    ax_cat.set_ylabel("Growth rate 1997-2016 (%)")
    ax_cat.set_title("Growth rate vs average GDHI (tertiles)", fontweight="bold")
    ax_cat.grid(True, alpha=0.3)
    ax_cat.legend(title="Income tier")
    fig_cat.tight_layout()
    fig_cat.savefig(OUTPUT_DIR / "week6_multivariate_growth.png", dpi=300, bbox_inches="tight")
    figs.append(fig_cat)
    if verbose:
        print("[OK] Saved: week6_multivariate_growth.png")

    return figs


def linear_regression_model(
    df: pd.DataFrame, verbose: bool = True
) -> tuple[LinearRegression, Figure, dict[str, float]]:
    X_reg = df[["avg_gdhi", "growth_rate", "volatility"]]
    y_reg = df["2016"]
    mask = X_reg.notna().all(axis=1) & y_reg.notna()
    X_train, X_test, y_train, y_test = train_test_split(X_reg[mask], y_reg[mask], test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("PART 4: LINEAR REGRESSION - PREDICTING 2016 GDHI")
        print("=" * 70)
        print("Training R^2: {:.4f}".format(metrics["train_r2"]))
        print("Testing  R^2: {:.4f}".format(metrics["test_r2"]))
        print("Training RMSE: GBP {:.2f}".format(metrics["train_rmse"]))
        print("Testing  RMSE: GBP {:.2f}".format(metrics["test_rmse"]))
        print("\nFeature coefficients:")
        for feature, coef in zip(X_reg.columns, model.coef_):
            print(f"  {feature}: {coef:.4f}")
        print(f"  Intercept: {model.intercept_:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors="black")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual GDHI 2016 (GBP)")
    axes[0].set_ylabel("Predicted GDHI 2016 (GBP)")
    axes[0].set_title(f"Linear regression fit (R^2 = {metrics['test_r2']:.4f})", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuals, alpha=0.6, edgecolors="black")
    axes[1].axhline(y=0, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted GDHI 2016 (GBP)")
    axes[1].set_ylabel("Residuals (GBP)")
    axes[1].set_title("Residual plot", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if verbose:
        fig.savefig(OUTPUT_DIR / "week6_linear_regression.png", dpi=300, bbox_inches="tight")
        print("[OK] Saved: week6_linear_regression.png")
    return model, fig, metrics


def kmeans_clustering(
    df: pd.DataFrame, features: list[str], verbose: bool = True
) -> tuple[pd.DataFrame, Figure, Figure, dict[str, float]]:
    X_cluster = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    inertias: List[float] = []
    silhouettes: List[float] = []
    k_values = range(2, 11)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, km.labels_)))

    fig_elbow, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_values, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_values, silhouettes, "ro-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette analysis", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    fig_elbow.tight_layout()
    fig_elbow.savefig(OUTPUT_DIR / "week6_kmeans_diagnostics.png", dpi=300, bbox_inches="tight")

    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_cluster = df.loc[X_cluster.index].copy()
    df_cluster["cluster"] = clusters

    fig_clusters, axes_cluster = plt.subplots(1, 2, figsize=(14, 6))
    axes_cluster[0].scatter(
        df_cluster["avg_gdhi"],
        df_cluster["growth_rate"],
        c=df_cluster["cluster"],
        cmap="tab10",
        alpha=0.75,
        edgecolors="black",
    )
    axes_cluster[0].set_xlabel("Average GDHI (GBP)")
    axes_cluster[0].set_ylabel("Growth rate 1997-2016 (%)")
    axes_cluster[0].set_title("Clusters: average GDHI vs growth", fontweight="bold")
    axes_cluster[0].grid(True, alpha=0.3)

    axes_cluster[1].scatter(
        df_cluster["volatility"],
        df_cluster["period_change"],
        c=df_cluster["cluster"],
        cmap="tab10",
        alpha=0.75,
        edgecolors="black",
    )
    axes_cluster[1].set_xlabel("Volatility (standard deviation, GBP)")
    axes_cluster[1].set_ylabel("Period change (GBP)")
    axes_cluster[1].set_title("Clusters: volatility vs period change", fontweight="bold")
    axes_cluster[1].grid(True, alpha=0.3)

    fig_clusters.tight_layout()
    fig_clusters.savefig(OUTPUT_DIR / "week6_kmeans_clusters.png", dpi=300, bbox_inches="tight")

    silhouette_best = silhouette_score(X_scaled, clusters)
    cluster_metrics = {"optimal_k": optimal_k, "silhouette": silhouette_best}
    if verbose:
        print("\n" + "=" * 70)
        print("PART 5: K-MEANS CLUSTERING")
        print("=" * 70)
        print(f"Best k (selected): {optimal_k}")
        print(f"Silhouette score: {silhouette_best:.3f}")
        print("[OK] Saved: week6_kmeans_diagnostics.png")
        print("[OK] Saved: week6_kmeans_clusters.png")

    return df_cluster, fig_elbow, fig_clusters, cluster_metrics


def knn_classification(
    df_cluster: pd.DataFrame, verbose: bool = True
) -> tuple[KNeighborsClassifier, Figure, Figure, dict[str, float]]:
    df_cluster = df_cluster.copy()
    median_growth = df_cluster["growth_rate"].median()
    df_cluster["growth_class"] = (df_cluster["growth_rate"] > median_growth).astype(int)
    df_cluster["growth_label"] = df_cluster["growth_class"].map({0: "Low growth", 1: "High growth"})

    if verbose:
        print("\n" + "=" * 70)
        print("PART 6: KNN CLASSIFICATION - GROWTH CATEGORY")
        print("=" * 70)
        print(df_cluster["growth_label"].value_counts())

    X_knn = df_cluster[["avg_gdhi", "volatility", "period_change"]]
    y_knn = df_cluster["growth_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_knn, y_knn, test_size=0.2, random_state=42, stratify=y_knn
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k_values = range(1, 21)
    train_acc: List[float] = []
    test_acc: List[float] = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        train_acc.append(float(knn.score(X_train_scaled, y_train)))
        test_acc.append(float(knn.score(X_test_scaled, y_test)))

    fig_k, ax_k = plt.subplots(figsize=(10, 6))
    ax_k.plot(k_values, train_acc, "bo-", label="Training accuracy", linewidth=2)
    ax_k.plot(k_values, test_acc, "ro-", label="Testing accuracy", linewidth=2)
    ax_k.set_xlabel("Number of neighbours (k)")
    ax_k.set_ylabel("Accuracy")
    ax_k.set_title("KNN accuracy by k", fontweight="bold")
    ax_k.legend()
    ax_k.grid(True, alpha=0.3)
    fig_k.tight_layout()
    fig_k.savefig(OUTPUT_DIR / "week6_knn_k_selection.png", dpi=300, bbox_inches="tight")

    optimal_k = test_acc.index(max(test_acc)) + 1
    knn_final = KNeighborsClassifier(n_neighbors=optimal_k).fit(X_train_scaled, y_train)
    y_pred = knn_final.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "optimal_k": optimal_k,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("PART 7: CLASSIFICATION PERFORMANCE")
        print("=" * 70)
        for name, value in metrics.items():
            if name == "optimal_k":
                continue
            print(f"{name.capitalize():>10}: {value:.4f} ({value*100:.2f}%)")

    cm = confusion_matrix(y_test, y_pred)
    fig_metrics, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Low growth", "High growth"],
        yticklabels=["Low growth", "High growth"],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_title("Confusion matrix", fontweight="bold")

    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    metric_values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
    bars = axes[1].bar(metric_labels, metric_values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Classification metrics", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, value in zip(bars, metric_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontweight="bold")

    fig_metrics.tight_layout()
    fig_metrics.savefig(OUTPUT_DIR / "week6_classification_metrics.png", dpi=300, bbox_inches="tight")

    if verbose:
        print("\nConfusion matrix:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["Low growth", "High growth"]))
        print("[OK] Saved: week6_knn_k_selection.png")
        print("[OK] Saved: week6_classification_metrics.png")

    return knn_final, fig_k, fig_metrics, metrics


def explain_precision_recall(metrics: dict[str, float], verbose: bool = True) -> None:
    if not verbose:
        return
    precision = metrics["precision"]
    recall = metrics["recall"]
    print("\n" + "=" * 70)
    print("PART 8: PRECISION VS RECALL - PRACTICAL CONTEXT")
    print("=" * 70)
    print(
        f"""
WHEN TO PRIORITISE PRECISION
----------------------------
- Investment decisions: avoid backing regions that will not grow (false positives are costly).
- Premium service targeting: ensure high-value services reach the right regions.
  Current precision: {precision:.3f} ({precision*100:.1f}% of predicted high-growth regions are correct.)

WHEN TO PRIORITISE RECALL
-------------------------
- Economic support programmes: missing a region in need is worse than reviewing an extra one.
- Market opportunity detection: better to investigate additional leads than miss top prospects.
  Current recall: {recall:.3f} (captures {recall*100:.1f}% of actual high-growth regions.)

Trade-off assessment: {"balanced" if abs(precision - recall) < 0.1 else ("precision-favoured" if precision > recall else "recall-favoured")}
"""
    )


def model_selection_summary(
    lr_metrics: dict[str, float], cluster_metrics: dict[str, float], knn_metrics: dict[str, float]
) -> None:
    print("\n" + "=" * 70)
    print("PART 9: MODEL SELECTION SUMMARY")
    print("=" * 70)
    comparison = pd.DataFrame(
        {
            "Model": ["Linear regression", "K-Means clustering", "KNN classification"],
            "Task": ["Regression", "Unsupervised", "Classification"],
            "Key metric": [
                f"R^2={lr_metrics['test_r2']:.3f}",
                f"Silhouette={cluster_metrics['silhouette']:.3f}",
                f"Accuracy={knn_metrics['accuracy']:.3f}",
            ],
            "Use when": [
                "Predicting future GDHI values",
                "Segmenting similar regions",
                "Labelling new regions by growth potential",
            ],
        }
    )
    print(comparison.to_string(index=False))


def summary_takeaways(
    lr_metrics: dict[str, float],
    cluster_metrics: dict[str, float],
    knn_metrics: dict[str, float],
) -> None:
    print("\n" + "=" * 70)
    print("WEEK 6 ANALYSIS - KEY TAKEAWAYS")
    print("=" * 70)
    print(
        f"""
FEATURE SELECTION
- Eight derived indicators summarise the GDHI time-series, covering level, growth, and volatility.
- Four predictors (avg_gdhi, growth_rate, volatility, period_change) drive the modelling workflows.

UNIVARIATE INSIGHTS
- Distributions highlight wide regional gaps and skewness in income changes.
- Visual dashboards map the spread and highlight median vs mean deviations.

MULTIVARIATE INSIGHTS
- Scatter matrix confirms strong associations among engineered features.
- Grouping by income tertiles shows clear separation of growth dynamics.

MODEL PERFORMANCE
- Linear regression achieves R^2={lr_metrics['test_r2']:.3f} with RMSE=GBP {lr_metrics['test_rmse']:.2f}.
- K-Means (k={cluster_metrics['optimal_k']}) delivers silhouette={cluster_metrics['silhouette']:.3f}.
- KNN reaches accuracy={knn_metrics['accuracy']:.3f}, precision={knn_metrics['precision']:.3f}, recall={knn_metrics['recall']:.3f}.

BUSINESS VALUE
- Regression supports forward income forecasts.
- Clustering reveals archetypes for policy or commercial focus.
- Classification flags regions to prioritise for growth initiatives.

Outputs saved under the outputs/ directory.
"""
    )


def prepare_outputs(path: str | None, verbose: bool = True) -> Tuple[List[Tuple[str, Figure]], str, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df, actual_path = load_data(path, verbose=True)
        engineered, selected, fig_corr = engineer_features(df, verbose=True)
        fig_uni = univariate_analysis(engineered, selected, verbose=True)
        multi_figs = multivariate_analysis(engineered, selected, verbose=True)
        _, fig_lr, lr_metrics = linear_regression_model(engineered, verbose=True)
        df_cluster, fig_elbow, fig_clusters, cluster_metrics = kmeans_clustering(engineered, selected, verbose=True)
        _, fig_k, fig_knn, knn_metrics = knn_classification(df_cluster, verbose=True)
        explain_precision_recall(knn_metrics, verbose=True)
        model_selection_summary(lr_metrics, cluster_metrics, knn_metrics)
        summary_takeaways(lr_metrics, cluster_metrics, knn_metrics)

    summary = buffer.getvalue()
    if verbose:
        print(summary)

    figures: List[Tuple[str, Figure]] = [
        ("Feature Correlation", fig_corr),
        ("Univariate Distributions", fig_uni),
    ]
    for idx, fig in enumerate(multi_figs, start=1):
        figures.append((f"Multivariate Analysis {idx}", fig))
    figures.append(("Linear Regression", fig_lr))
    figures.append(("K-Means Diagnostics", fig_elbow))
    figures.append(("K-Means Clusters", fig_clusters))
    figures.append(("KNN Accuracy vs k", fig_k))
    figures.append(("KNN Metrics", fig_knn))

    return figures, summary, actual_path


def run(config: dict) -> None:
    import streamlit as st  # type: ignore

    figures, summary, actual_path = prepare_outputs(config.get("data_path"), verbose=False)
    st.header("Week 6 - Machine learning with GDHI data")
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
