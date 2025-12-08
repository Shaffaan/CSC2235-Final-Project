import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import polars as pl


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from common.download_utils import (  # noqa: E402
    download_openimages_class_descriptions,
    download_openimages_detection,
)


AREA_BUCKET_EDGES = [0.0, 0.02, 0.15, float("inf")]
AREA_BUCKET_LABELS = ["small", "medium", "large"]
SIZE_PRESET_LIMITS = {
    "tiny": 50_000,
    "small": 200_000,
    "medium": 1_000_000,
    "large": 5_000_000,
    "xlarge": 10_000_000,
}


@dataclass
class OpenImagesConfig:
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "cv_openimages")
    detection_filename: str = "oidv6-train-annotations-bbox.csv"
    class_filename: str = "oidv6-class-descriptions.csv"
    limit: Optional[int] = None
    results_dir: Optional[str] = None

    def resolve_results_dir(self) -> str:
        if self.results_dir:
            return self.results_dir
        env_dir = os.environ.get("SCRIPT_RESULTS_DIR")
        if env_dir:
            return env_dir
        return os.path.join(PROJECT_ROOT, "results", "local_test", "cv", "openimages_polars")


def ensure_openimages_assets(config: OpenImagesConfig) -> Tuple[str, str]:
    annotations_path = download_openimages_detection(config.data_dir, config.detection_filename)
    class_map_path = download_openimages_class_descriptions(config.data_dir, config.class_filename)
    return annotations_path, class_map_path


def load_detections(csv_path: str, limit: Optional[int]) -> pl.DataFrame:
    print(f"--- Loading detection annotations from {csv_path} ---")
    df = pl.read_csv(csv_path, n_rows=limit)
    print(f"  Loaded {len(df):,} detection rows")
    return df


def load_class_descriptions(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    if not {"LabelName", "DisplayName"}.issubset(df.columns):
        raise KeyError("Expected columns 'LabelName' and 'DisplayName' in class descriptions")
    return df


def compute_box_metrics(df: pl.DataFrame) -> pl.DataFrame:
    required = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required detection columns: {', '.join(sorted(missing))}")

    df = df.with_columns(
        [
            pl.col(column).cast(pl.Float64)
            for column in ["XMin", "XMax", "YMin", "YMax"]
            if column in df.columns
        ]
    )

    df = df.with_columns(
        [
            (pl.col("XMax") - pl.col("XMin")).clip(0.0, None).alias("box_width"),
            (pl.col("YMax") - pl.col("YMin")).clip(0.0, None).alias("box_height"),
        ]
    )
    df = df.with_columns((pl.col("box_width") * pl.col("box_height")).clip(0.0, None).alias("box_area_ratio"))

    has_image_dims = {"ImageWidth", "ImageHeight"}.issubset(df.columns)
    if has_image_dims:
        df = df.with_columns(
            [
                pl.col("ImageWidth").cast(pl.Float64),
                pl.col("ImageHeight").cast(pl.Float64),
                (pl.col("ImageWidth") * pl.col("ImageHeight")).alias("image_area_px"),
            ]
        )
        df = df.with_columns(
            (
                pl.col("box_width")
                * pl.col("ImageWidth")
                * pl.col("box_height")
                * pl.col("ImageHeight")
            ).alias("box_area_px")
        )
        df = df.with_columns(
            (
                pl.col("box_area_px")
                / pl.when(pl.col("image_area_px") == 0.0)
                .then(None)
                .otherwise(pl.col("image_area_px"))
            )
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("normalized_area")
        )
    else:
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("image_area_px"),
                pl.lit(None, dtype=pl.Float64).alias("box_area_px"),
                pl.col("box_area_ratio").alias("normalized_area"),
            ]
        )

    df = df.with_columns(pl.col("normalized_area").clip(0.0, 1.0).alias("normalized_area"))
    size_bucket_expr = (
        pl.when(pl.col("normalized_area") < AREA_BUCKET_EDGES[1])
        .then(pl.lit(AREA_BUCKET_LABELS[0]))
        .when(pl.col("normalized_area") < AREA_BUCKET_EDGES[2])
        .then(pl.lit(AREA_BUCKET_LABELS[1]))
        .otherwise(pl.lit(AREA_BUCKET_LABELS[2]))
    )
    df = df.with_columns(size_bucket_expr.alias("size_bucket"))

    return df


def attach_class_names(detections: pl.DataFrame, class_df: pl.DataFrame) -> pl.DataFrame:
    merged = detections.join(class_df, on="LabelName", how="left")
    return merged.rename({"DisplayName": "class_name", "LabelName": "class_id"})


def summarize_class_areas(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    area_stats = (
        df.group_by(["class_id", "class_name"], maintain_order=False)
        .agg(
            pl.len().alias("detections"),
            pl.col("normalized_area").mean().alias("mean_norm_area"),
            pl.col("normalized_area").median().alias("median_norm_area"),
            pl.col("normalized_area").std().alias("std_norm_area"),
        )
        .sort("detections", descending=True)
    )

    bucket_counts = (
        df.group_by(["class_id", "class_name", "size_bucket"], maintain_order=False)
        .agg(pl.len().alias("detections"))
        .sort(["class_id", "size_bucket"])
    )

    return area_stats, bucket_counts


def summarize_bucket_distribution(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("size_bucket", maintain_order=False)
        .agg(
            pl.len().alias("detections"),
            pl.col("normalized_area").mean().alias("avg_normalized_area"),
        )
        .sort("size_bucket")
    )


def persist_results(
    results_dir: str,
    area_stats: pl.DataFrame,
    bucket_counts: pl.DataFrame,
    bucket_summary: pl.DataFrame,
) -> Dict[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    outputs = {
        "class_area_stats": os.path.join(results_dir, "class_area_stats.csv"),
        "class_bucket_counts": os.path.join(results_dir, "class_bucket_counts.csv"),
        "overall_bucket_summary": os.path.join(results_dir, "bucket_summary.csv"),
    }
    area_stats.write_csv(outputs["class_area_stats"])
    bucket_counts.write_csv(outputs["class_bucket_counts"])
    bucket_summary.write_csv(outputs["overall_bucket_summary"])
    return outputs


def _record_timing(
    timings: List[Dict[str, Any]],
    step: str,
    start_time: float,
    **metadata: Any,
) -> None:
    duration = time.perf_counter() - start_time
    record: Dict[str, Any] = {"step": step, "duration_s": duration}
    record.update(metadata)
    timings.append(record)


def persist_timings(timings: List[Dict[str, Any]], results_dir: str) -> Optional[str]:
    if not timings:
        return None
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "step_timings.csv")
    pl.DataFrame(timings).write_csv(path)
    return path


def run_openimages_pipeline(config: OpenImagesConfig) -> Dict[str, Any]:
    timings: List[Dict[str, Any]] = []

    annotations_path, class_map_path = ensure_openimages_assets(config)

    start = time.perf_counter()
    detections = load_detections(annotations_path, config.limit)
    _record_timing(timings, "load_detections", start, rows=len(detections))

    start = time.perf_counter()
    detections = compute_box_metrics(detections)
    _record_timing(timings, "vector_math_box_metrics", start, rows=len(detections))

    start = time.perf_counter()
    class_df = load_class_descriptions(class_map_path)
    _record_timing(timings, "load_class_map", start, classes=len(class_df))

    start = time.perf_counter()
    enriched = attach_class_names(detections, class_df)
    _record_timing(timings, "join_class_names", start, rows=len(enriched))

    start = time.perf_counter()
    area_stats, bucket_counts = summarize_class_areas(enriched)
    _record_timing(
        timings,
        "group_area_by_class",
        start,
        classes=len(area_stats),
        bucket_rows=len(bucket_counts),
    )

    start = time.perf_counter()
    bucket_summary = summarize_bucket_distribution(enriched)
    _record_timing(timings, "group_bucket_distribution", start, buckets=len(bucket_summary))

    results_dir = config.resolve_results_dir()
    start = time.perf_counter()
    outputs = persist_results(results_dir, area_stats, bucket_counts, bucket_summary)
    _record_timing(timings, "persist_results", start)

    timings_path = persist_timings(timings, results_dir)
    print("--- Wrote analysis artifacts ---")
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    if timings_path:
        print(f"  step_timings: {timings_path}")

    print("\nSample class-level stats:")
    print(area_stats.head(10))

    print("\nOverall bucket distribution:")
    print(bucket_summary)

    print("\nStep timings (seconds):")
    print(pl.DataFrame(timings))

    return {
        "outputs": outputs,
        "area_stats": area_stats,
        "bucket_counts": bucket_counts,
        "bucket_summary": bucket_summary,
        "rows_processed": len(enriched),
        "results_dir": results_dir,
        "timings": timings,
        "timings_csv": timings_path,
    }


def parse_args() -> Tuple[OpenImagesConfig, Optional[str]]:
    parser = argparse.ArgumentParser(
        description="Open Images bounding box statistics with Polars"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data", "cv_openimages"),
        help="Directory to store/download Open Images artifacts.",
    )
    parser.add_argument(
        "--annotation-file",
        default="oidv6-train-annotations-bbox.csv",
        help="Detection annotations CSV filename to download/use.",
    )
    parser.add_argument(
        "--class-file",
        default="oidv6-class-descriptions.csv",
        help="Class description CSV filename to download/use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for loading annotations (useful for sampling).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to store CSV summaries (defaults to results/local_test/cv/openimages_polars).",
    )
    parser.add_argument(
        "--presets",
        default=None,
        help="Comma-separated preset names to run when --limit is omitted (default: all presets).",
    )
    args = parser.parse_args()
    config = OpenImagesConfig(
        data_dir=args.data_dir,
        detection_filename=args.annotation_file,
        class_filename=args.class_file,
        limit=args.limit,
        results_dir=args.results_dir,
    )
    return config, args.presets


def build_preset_configurations(
    base_config: OpenImagesConfig,
    preset_filter: Optional[str],
) -> List[Tuple[str, OpenImagesConfig]]:
    if not SIZE_PRESET_LIMITS:
        return []

    if preset_filter:
        requested = [name.strip() for name in preset_filter.split(",") if name.strip()]
    else:
        requested = list(SIZE_PRESET_LIMITS.keys())

    invalid = [name for name in requested if name not in SIZE_PRESET_LIMITS]
    if invalid:
        raise KeyError(
            f"Unknown preset(s): {', '.join(invalid)}. Available presets: {', '.join(SIZE_PRESET_LIMITS.keys())}"
        )

    base_dir = base_config.resolve_results_dir()
    configs: List[Tuple[str, OpenImagesConfig]] = []
    for name in requested:
        limit = SIZE_PRESET_LIMITS[name]
        preset_config = OpenImagesConfig(
            data_dir=base_config.data_dir,
            detection_filename=base_config.detection_filename,
            class_filename=base_config.class_filename,
            limit=limit,
            results_dir=os.path.join(base_dir, name),
        )
        configs.append((name, preset_config))
    return configs


def compute_size_bucket_shares(bucket_summary: pl.DataFrame) -> Dict[str, float]:
    if bucket_summary.is_empty():
        return {f"{bucket}_share": 0.0 for bucket in AREA_BUCKET_LABELS}

    bucket_counts = dict(
        zip(bucket_summary["size_bucket"].to_list(), bucket_summary["detections"].to_list())
    )
    total = float(bucket_summary["detections"].sum()) or 1.0
    shares = {}
    for bucket in AREA_BUCKET_LABELS:
        count = float(bucket_counts.get(bucket, 0.0))
        shares[f"{bucket}_share"] = count / total
    return shares


def persist_preset_summary(records: List[Dict[str, Any]], summary_dir: str) -> Optional[str]:
    if not records:
        return None

    os.makedirs(summary_dir, exist_ok=True)
    summary_df = pl.DataFrame(records)
    summary_path = os.path.join(summary_dir, "openimages_polars_preset_summary.csv")
    summary_df.write_csv(summary_path)
    print("\nPreset summary table:")
    print(summary_df)
    return summary_path


def main() -> None:
    base_config, preset_filter = parse_args()

    if base_config.limit is not None:
        run_openimages_pipeline(base_config)
        return

    preset_runs = build_preset_configurations(base_config, preset_filter)
    if not preset_runs:
        print(
            "No size presets defined and --limit was not provided. Please set a row limit or define presets."
        )
        return

    summary_records: List[Dict[str, Any]] = []
    for preset_name, preset_config in preset_runs:
        print(f"\n=== Running preset '{preset_name}' (row limit={preset_config.limit:,}) ===")
        run_result = run_openimages_pipeline(preset_config)
        record = {
            "preset": preset_name,
            "row_limit": preset_config.limit,
            "rows_processed": run_result["rows_processed"],
            "results_dir": run_result["results_dir"],
        }
        record.update(compute_size_bucket_shares(run_result["bucket_summary"]))
        summary_records.append(record)

    summary_path = persist_preset_summary(summary_records, base_config.resolve_results_dir())
    if summary_path:
        print(f"\nPreset summary CSV saved to {summary_path}")


if __name__ == "__main__":
    main()
