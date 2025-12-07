import glob
import os
import sys
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.io as pio


STATIC_FRAMEWORKS: Dict[str, Tuple[str, str]] = {
    "pandas": ("pandas", "join_metrics_*.csv"),
    "polars": ("polars", "join_metrics_*.csv"),
    "duckdb": ("duckdb", "join_metrics_*.csv"),
}
SPARK_PATTERN = "spark_join_metrics_*.csv"


def _iter_framework_sources(results_root: str) -> Sequence[Tuple[str, str, str]]:
    """
    Generates (framework_label, framework_dir, glob_pattern) tuples for summary discovery.
    """
    sources: List[Tuple[str, str, str]] = []

    for label, (subdir, pattern) in STATIC_FRAMEWORKS.items():
        sources.append((label, os.path.join(results_root, subdir), pattern))

    for entry in sorted(os.listdir(results_root)):
        if not entry.startswith("spark"):
            continue
        framework_dir = os.path.join(results_root, entry)
        if not os.path.isdir(framework_dir):
            continue
        sources.append((entry, framework_dir, SPARK_PATTERN))

    return sources


def _load_join_metrics(results_root: str) -> pd.DataFrame:
    """
    Reads every join_metrics_*.csv that each framework emitted and returns a combined DataFrame.
    """
    frames: List[pd.DataFrame] = []

    for framework, framework_dir, pattern in _iter_framework_sources(results_root):
        glob_pattern = os.path.join(framework_dir, pattern)
        csv_files = glob.glob(glob_pattern)

        if not csv_files:
            print(f"[summarize] No join metrics found for {framework} under {framework_dir}")
            continue

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            df["framework"] = framework
            df["source_file"] = os.path.relpath(csv_path, results_root)
            frames.append(df)
            print(f"[summarize] Loaded {len(df)} rows from {csv_path}")

    if not frames:
        raise FileNotFoundError(f"No join metrics found under {results_root}. Did you run the pipelines?")

    combined = pd.concat(frames, ignore_index=True)

    numeric_cols = [
        "left_rows",
        "right_rows",
        "result_rows",
        "columns_returned",
        "execution_time_s",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


def _ensure_output_dirs(results_root: str) -> Dict[str, str]:
    plots_dir = os.path.join(results_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return {"plots": plots_dir}


def _aggregate_join_metrics(join_df: pd.DataFrame, plots_dir: str):
    """
    Creates per-join-type visualizations and summary tables.
    """
    pio.templates.default = "plotly_dark"

    summary = (
        join_df.groupby(["config_name", "join_type", "framework"])
        .agg(
            avg_execution_time_s=("execution_time_s", "mean"),
            median_execution_time_s=("execution_time_s", "median"),
            max_execution_time_s=("execution_time_s", "max"),
            min_execution_time_s=("execution_time_s", "min"),
            run_count=("execution_time_s", "count"),
        )
        .reset_index()
    )

    page_rows = (
        join_df[join_df["left_table"] == "pages"]
        .groupby("config_name")["left_rows"]
        .max()
        .rename("page_rows")
        .reset_index()
    )

    summary = summary.merge(page_rows, on="config_name", how="left")

    summary_path = os.path.join(plots_dir, "wikipedia_join_type_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[summarize] Saved summary table to {summary_path}")

    size_order = {"tiny": 0, "small": 1, "medium": 2, "large": 3}

    def sort_key(config_name: str) -> float:
        lowered = config_name.lower()
        for size, idx in size_order.items():
            if size in lowered:
                return idx + hash(config_name) * 0.0
        return len(size_order)

    config_order = sorted(join_df["config_name"].dropna().unique(), key=sort_key)
    page_rows_map = {
        row["config_name"]: row["page_rows"]
        for _, row in summary[["config_name", "page_rows"]].drop_duplicates(subset=["config_name"]).iterrows()
    }

    for join_type in sorted(join_df["join_type"].dropna().unique()):
        subset = summary[summary["join_type"] == join_type].copy()
        if subset.empty:
            continue
        subset["config_name"] = pd.Categorical(subset["config_name"], categories=config_order, ordered=True)
        subset = subset.sort_values("config_name")
        subset["config_axis"] = subset.apply(
            lambda row: (
                f"{row['config_name']} ({int(row['page_rows']):,} pages)"
                if pd.notna(row["page_rows"])
                else row["config_name"]
            ),
            axis=1,
        )
        axis_order = [
            f"{name} ({int(page_rows_map[name]):,} pages)"
            if name in page_rows_map and pd.notna(page_rows_map[name])
            else name
            for name in config_order
        ]

        fig = px.line(
            subset,
            x="config_axis",
            y="avg_execution_time_s",
            color="framework",
            markers=True,
            title=f"Average Execution Time by Framework — Join: {join_type}",
            labels={
                "avg_execution_time_s": "Avg Execution Time (s)",
                "config_name": "Configuration",
                "framework": "Framework",
            },
        )
        fig.update_layout(
            yaxis=dict(tickformat=".4f"),
            xaxis=dict(type="category", categoryorder="array", categoryarray=axis_order),
        )

        output_file = os.path.join(plots_dir, f"join_type_{join_type}_execution_time.html")
        fig.write_html(output_file)
        print(f"[summarize] Wrote plot: {output_file}")


def _main():
    if len(sys.argv) != 2:
        print("Usage: python common/summarize_results_wikipedia.py <wikipedia_results_dir>")
        print("  Example: python common/summarize_results_wikipedia.py results/2025-11-21_22-31-23/wikipedia_data")
        sys.exit(1)

    results_root = os.path.abspath(sys.argv[1])
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    print(f"[summarize] Collecting Wikipedia join metrics from {results_root}")
    join_df = _load_join_metrics(results_root)
    dirs = _ensure_output_dirs(results_root)
    _aggregate_join_metrics(join_df, dirs["plots"])
    print("[summarize] Done.")


if __name__ == "__main__":
    _main()
