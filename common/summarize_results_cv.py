import os
import sys
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.io as pio


REQUIRED_FILES: Dict[str, str] = {
    "class_area_stats": "class_area_stats.csv",
    "class_bucket_counts": "class_bucket_counts.csv",
    "bucket_summary": "bucket_summary.csv",
}


def _has_run_artifacts(path: str) -> bool:
    return all(os.path.isfile(os.path.join(path, filename)) for filename in REQUIRED_FILES.values())


def _discover_pipeline_dirs(results_root: str) -> Sequence[Tuple[str, str]]:
    if _has_run_artifacts(results_root):
        return [(os.path.basename(results_root.rstrip(os.sep)), results_root)]

    pipeline_dirs: List[Tuple[str, str]] = []
    for entry in sorted(os.listdir(results_root)):
        candidate = os.path.join(results_root, entry)
        if os.path.isdir(candidate):
            pipeline_dirs.append((entry, candidate))
    return pipeline_dirs


def _discover_run_dirs(pipeline_dir: str) -> Sequence[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    if _has_run_artifacts(pipeline_dir):
        runs.append(("default", pipeline_dir))

    for entry in sorted(os.listdir(pipeline_dir)):
        candidate = os.path.join(pipeline_dir, entry)
        if not os.path.isdir(candidate):
            continue
        if _has_run_artifacts(candidate):
            runs.append((entry, candidate))

    return runs


def _load_run_artifacts(pipeline: str, run_label: str, run_dir: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for key, filename in REQUIRED_FILES.items():
        csv_path = os.path.join(run_dir, filename)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing {filename} in {run_dir}")
        df = pd.read_csv(csv_path)
        df["pipeline"] = pipeline
        df["run_label"] = run_label
        data[key] = df
        print(f"[summarize-cv] Loaded {len(df)} rows from {csv_path}")
    return data


def _concat_or_none(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_csv(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[summarize-cv] Saved {len(df)} rows to {path}")


def _print_top_classes(area_df: pd.DataFrame) -> None:
    if area_df.empty:
        return

    for (pipeline, run_label), group_df in area_df.groupby(["pipeline", "run_label"]):
        top = group_df.nlargest(5, "detections")[
            ["class_name", "detections", "mean_norm_area", "median_norm_area"]
        ]
        print(f"\n[summarize-cv] Top classes for {pipeline}/{run_label}:")
        print(top.to_string(index=False))


def _compute_bucket_shares(bucket_df: pd.DataFrame) -> pd.DataFrame:
    if bucket_df.empty:
        return bucket_df

    bucket_df = bucket_df.copy()
    total = bucket_df.groupby(["pipeline", "run_label"])["detections"].transform("sum")
    bucket_df["detection_share"] = bucket_df["detections"] / total.replace({0: 1})
    return bucket_df


def _render_timing_plots(per_run: pd.DataFrame, summary_dir: str) -> None:
    if per_run.empty:
        return

    plots_dir = os.path.join(summary_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    pio.templates.default = "plotly_dark"

    per_run = per_run.sort_values(["pipeline", "rows_processed"])

    fig = px.line(
        per_run,
        x="rows_processed",
        y="total_duration_s",
        color="pipeline",
        markers=True,
        line_group="pipeline",
        hover_data={
            "pipeline": True,
            "run_label": True,
            "rows_processed": ":,",
            "total_duration_s": ".3f",
        },
        labels={
            "rows_processed": "Rows Processed",
            "total_duration_s": "Total Runtime (s)",
            "pipeline": "Pipeline",
            "run_label": "Run Label",
        },
        title="CV Pipeline Runtime vs Rows Processed",
    )

    fig.update_layout(xaxis=dict(tickformat=",d"), yaxis=dict(tickformat=".3f"))
    output_file = os.path.join(plots_dir, "runtime_vs_rows.html")
    fig.write_html(output_file)
    print(f"[summarize-cv] Wrote plot: {output_file}")


def _main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python common/summarize_results_cv.py <cv_results_dir>")
        print("  Example: python common/summarize_results_cv.py results/2025-03-01_12-00-00/cv")
        sys.exit(1)

    results_root = os.path.abspath(sys.argv[1])
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    pipeline_dirs = _discover_pipeline_dirs(results_root)
    if not pipeline_dirs:
        raise FileNotFoundError(f"No pipeline directories found under {results_root}")

    bucket_frames: List[pd.DataFrame] = []
    area_frames: List[pd.DataFrame] = []
    count_frames: List[pd.DataFrame] = []
    timing_frames: List[pd.DataFrame] = []

    for pipeline, pipeline_dir in pipeline_dirs:
        run_dirs = _discover_run_dirs(pipeline_dir)
        if not run_dirs:
            print(f"[summarize-cv] Skipping {pipeline}: no run directories with artifacts found")
            continue

        for run_label, run_dir in run_dirs:
            artifacts = _load_run_artifacts(pipeline, run_label, run_dir)
            bucket_df = artifacts["bucket_summary"]
            bucket_frames.append(bucket_df)
            area_frames.append(artifacts["class_area_stats"])
            count_frames.append(artifacts["class_bucket_counts"])

            rows_processed = int(bucket_df["detections"].sum()) if not bucket_df.empty else 0

            timing_path = os.path.join(run_dir, "step_timings.csv")
            if os.path.isfile(timing_path):
                timings_df = pd.read_csv(timing_path)
                timings_df["pipeline"] = pipeline
                timings_df["run_label"] = run_label
                timings_df["rows_processed"] = rows_processed
                timing_frames.append(timings_df)
                print(
                    f"[summarize-cv] Loaded {len(timings_df)} timing rows from {timing_path}"
                )

    bucket_df = _compute_bucket_shares(_concat_or_none(bucket_frames))
    area_df = _concat_or_none(area_frames)
    count_df = _concat_or_none(count_frames)
    timing_df = _concat_or_none(timing_frames)

    if bucket_df.empty and area_df.empty and count_df.empty:
        raise FileNotFoundError(
            f"No CV artifacts found under {results_root}. Did you run any CV pipelines?"
        )

    summary_dir = os.path.join(results_root, "summaries")
    _write_csv(bucket_df, os.path.join(summary_dir, "cv_bucket_summary_combined.csv"))
    _write_csv(area_df, os.path.join(summary_dir, "cv_class_area_stats_combined.csv"))
    _write_csv(count_df, os.path.join(summary_dir, "cv_class_bucket_counts_combined.csv"))
    _write_csv(timing_df, os.path.join(summary_dir, "cv_step_timings_combined.csv"))

    if not timing_df.empty:
        per_run = (
            timing_df.groupby(["pipeline", "run_label", "rows_processed"], dropna=False)[
                "duration_s"
            ]
            .sum()
            .reset_index()
            .rename(columns={"duration_s": "total_duration_s"})
            .sort_values("rows_processed")
        )
        _write_csv(per_run, os.path.join(summary_dir, "cv_run_timing_summary.csv"))
        _render_timing_plots(per_run, summary_dir)
        print("\n[summarize-cv] Total pipeline runtimes (s):")
        print(per_run.to_string(index=False))

    _print_top_classes(area_df)
    print("\n[summarize-cv] Done.")


if __name__ == "__main__":
    _main()