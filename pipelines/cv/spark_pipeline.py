import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F


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
    "large": 10_000_000,
}


@dataclass
class OpenImagesConfig:
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "cv_openimages")
    detection_filename: str = "oidv6-train-annotations-bbox.csv"
    class_filename: str = "oidv6-class-descriptions.csv"
    limit: Optional[int] = None
    results_dir: Optional[str] = None
    spark_master: Optional[str] = None
    spark_executor_memory: str = "8g"
    spark_driver_memory: str = "4g"

    def resolve_results_dir(self) -> str:
        if self.results_dir:
            return self.results_dir
        env_dir = os.environ.get("SCRIPT_RESULTS_DIR")
        if env_dir:
            return env_dir
        return os.path.join(PROJECT_ROOT, "results", "local_test", "cv", "openimages_spark")


_EFFECTIVE_SPARK_MODE = os.environ.get("SPARK_MODE", "local").strip().lower()


def ensure_openimages_assets(config: OpenImagesConfig) -> Tuple[str, str]:
    annotations_path = download_openimages_detection(config.data_dir, config.detection_filename)
    class_map_path = download_openimages_class_descriptions(config.data_dir, config.class_filename)
    return annotations_path, class_map_path


def _resolve_spark_mode_and_master(config: OpenImagesConfig) -> Tuple[str, str]:
    global _EFFECTIVE_SPARK_MODE
    requested_mode = os.environ.get("SPARK_MODE", "local").strip().lower()
    env_master = os.environ.get("SPARK_MASTER_URL")
    explicit_master = config.spark_master or env_master

    if not explicit_master and env_master and env_master.startswith("spark://"):
        explicit_master = env_master

    resolved_mode = requested_mode
    if resolved_mode != "distributed" and explicit_master and explicit_master.startswith("spark://"):
        resolved_mode = "distributed"

    if resolved_mode == "distributed":
        if not explicit_master:
            raise RuntimeError(
                "SPARK_MODE=distributed requires a Spark master URL. "
                "Set --spark-master or the SPARK_MASTER_URL environment variable."
            )
        master = explicit_master
    else:
        if explicit_master:
            master = explicit_master
        else:
            local_threads = os.environ.get("SPARK_LOCAL_THREADS", "*")
            master = f"local[{local_threads}]"
        resolved_mode = "local"

    _EFFECTIVE_SPARK_MODE = resolved_mode
    return resolved_mode, master


def _resolve_input_path(path: str) -> str:
    """Adds a file:// prefix for local runs so Spark can read host-local files."""
    if "://" in path:
        return path
    spark_mode = _EFFECTIVE_SPARK_MODE
    abs_path = os.path.abspath(path)
    if spark_mode == "distributed":
        return abs_path
    return f"file://{abs_path}"


def create_spark_session(config: OpenImagesConfig) -> SparkSession:
    spark_mode, master = _resolve_spark_mode_and_master(config)
    app_name = f"OpenImagesSparkPipeline_{spark_mode}"

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", config.spark_driver_memory)
        .config("spark.executor.memory", config.spark_executor_memory)
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "200"))
    )

    if spark_mode == "distributed":
        print(f"  [Distributed] Connecting Spark session to {master}")
        distributed_cores = os.environ.get("SPARK_DISTRIBUTED_MAX_CORES")
        if distributed_cores:
            builder = builder.config("spark.cores.max", distributed_cores)
    else:
        print(f"  [Local] Starting Spark session with master={master}")

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session


def load_detections(spark: SparkSession, csv_path: str, limit: Optional[int]) -> Tuple[DataFrame, int]:
    print(f"--- Loading detection annotations from {csv_path} ---")
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(_resolve_input_path(csv_path))
    )
    if limit is not None:
        df = df.limit(int(limit))
    count = df.count()
    print(f"  Loaded {count:,} detection rows")
    return df, count


def load_class_descriptions(spark: SparkSession, csv_path: str) -> Tuple[DataFrame, int]:
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(_resolve_input_path(csv_path))
    )
    if not {"LabelName", "DisplayName"}.issubset(df.columns):
        raise KeyError("Expected columns 'LabelName' and 'DisplayName' in class descriptions")
    count = df.count()
    return df, count


def compute_box_metrics(df: DataFrame) -> DataFrame:
    required = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required detection columns: {', '.join(sorted(missing))}")

    df = df.select("*")
    numeric_cols = ["XMin", "XMax", "YMin", "YMax", "ImageWidth", "ImageHeight"]
    for column in numeric_cols:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast("double"))

    df = df.withColumn("box_width", F.greatest(F.col("XMax") - F.col("XMin"), F.lit(0.0)))
    df = df.withColumn("box_height", F.greatest(F.col("YMax") - F.col("YMin"), F.lit(0.0)))
    df = df.withColumn(
        "box_area_ratio",
        F.greatest(F.col("box_width") * F.col("box_height"), F.lit(0.0)),
    )

    has_image_dims = {"ImageWidth", "ImageHeight"}.issubset(df.columns)
    if has_image_dims:
        df = df.withColumn("image_area_px", F.col("ImageWidth") * F.col("ImageHeight"))
        df = df.withColumn(
            "box_area_px",
            F.col("box_width") * F.col("ImageWidth") * F.col("box_height") * F.col("ImageHeight"),
        )
        df = df.withColumn(
            "normalized_area",
            F.when((F.col("image_area_px").isNull()) | (F.col("image_area_px") == 0), F.col("box_area_ratio"))
            .otherwise(F.col("box_area_px") / F.col("image_area_px")),
        )
    else:
        df = df.withColumn("image_area_px", F.lit(None).cast("double"))
        df = df.withColumn("box_area_px", F.lit(None).cast("double"))
        df = df.withColumn("normalized_area", F.col("box_area_ratio"))

    df = df.withColumn(
        "normalized_area",
        F.when(F.col("normalized_area").isNull(), F.lit(0.0)).otherwise(F.col("normalized_area")),
    )
    df = df.withColumn(
        "normalized_area",
        F.when(F.col("normalized_area") < 0, F.lit(0.0))
        .when(F.col("normalized_area") > 1, F.lit(1.0))
        .otherwise(F.col("normalized_area")),
    )

    df = df.withColumn(
        "size_bucket",
        F.when(F.col("normalized_area") < AREA_BUCKET_EDGES[1], AREA_BUCKET_LABELS[0])
        .when(F.col("normalized_area") < AREA_BUCKET_EDGES[2], AREA_BUCKET_LABELS[1])
        .otherwise(AREA_BUCKET_LABELS[2]),
    )
    return df


def attach_class_names(detections: DataFrame, class_df: DataFrame) -> DataFrame:
    joined = detections.join(
        class_df.select("LabelName", "DisplayName"),
        detections["LabelName"] == class_df["LabelName"],
        "left",
    )
    joined = joined.drop(class_df["LabelName"])
    joined = joined.withColumnRenamed("LabelName", "class_id")
    joined = joined.withColumnRenamed("DisplayName", "class_name")
    return joined


def summarize_class_areas(df: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    area_stats_sdf = (
        df.groupBy("class_id", "class_name")
        .agg(
            F.count("*").alias("detections"),
            F.mean("normalized_area").alias("mean_norm_area"),
            F.expr("percentile_approx(normalized_area, 0.5)").alias("median_norm_area"),
            F.stddev_pop("normalized_area").alias("std_norm_area"),
        )
        .orderBy(F.col("detections").desc())
    )

    bucket_counts_sdf = (
        df.groupBy("class_id", "class_name", "size_bucket")
        .agg(F.count("*").alias("detections"))
        .orderBy("class_id", "size_bucket")
    )

    return area_stats_sdf.toPandas(), bucket_counts_sdf.toPandas()


def summarize_bucket_distribution(df: DataFrame) -> pd.DataFrame:
    bucket_summary_sdf = (
        df.groupBy("size_bucket")
        .agg(
            F.count("*").alias("detections"),
            F.mean("normalized_area").alias("avg_normalized_area"),
        )
        .orderBy("size_bucket")
    )
    return bucket_summary_sdf.toPandas()


def persist_results(
    results_dir: str,
    area_stats: pd.DataFrame,
    bucket_counts: pd.DataFrame,
    bucket_summary: pd.DataFrame,
) -> Dict[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    outputs = {
        "class_area_stats": os.path.join(results_dir, "class_area_stats.csv"),
        "class_bucket_counts": os.path.join(results_dir, "class_bucket_counts.csv"),
        "overall_bucket_summary": os.path.join(results_dir, "bucket_summary.csv"),
    }
    area_stats.to_csv(outputs["class_area_stats"], index=False)
    bucket_counts.to_csv(outputs["class_bucket_counts"], index=False)
    bucket_summary.to_csv(outputs["overall_bucket_summary"], index=False)
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
    pd.DataFrame(timings).to_csv(path, index=False)
    return path


def run_openimages_pipeline(config: OpenImagesConfig) -> Dict[str, Any]:
    timings: List[Dict[str, Any]] = []
    annotations_path, class_map_path = ensure_openimages_assets(config)
    spark = create_spark_session(config)
    try:
        start = time.perf_counter()
        detections_df, detection_count = load_detections(spark, annotations_path, config.limit)
        _record_timing(timings, "load_detections", start, rows=detection_count)

        start = time.perf_counter()
        class_df, class_count = load_class_descriptions(spark, class_map_path)
        _record_timing(timings, "load_class_map", start, classes=class_count)

        start = time.perf_counter()
        enriched = attach_class_names(compute_box_metrics(detections_df), class_df)
        enriched = enriched.cache()
        enriched_rows = enriched.count()
        _record_timing(timings, "compute_metrics_and_join", start, rows=enriched_rows)

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
        print(pd.DataFrame(timings))

        return {
            "outputs": outputs,
            "area_stats": area_stats,
            "bucket_counts": bucket_counts,
            "bucket_summary": bucket_summary,
            "rows_processed": enriched_rows,
            "results_dir": results_dir,
            "timings": timings,
            "timings_csv": timings_path,
        }
    finally:
        spark.stop()


def parse_args() -> Tuple[OpenImagesConfig, Optional[str]]:
    parser = argparse.ArgumentParser(description="Open Images bounding box statistics with Spark")
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
        help="Directory to store CSV summaries (defaults to results/local_test/cv/openimages_spark).",
    )
    parser.add_argument(
        "--presets",
        default=None,
        help="Comma-separated preset names to run when --limit is omitted (default: all presets).",
    )
    parser.add_argument(
        "--spark-master",
        default=None,
        help="Optional Spark master override (defaults to local[*] unless SPARK_MASTER_URL is set).",
    )
    parser.add_argument(
        "--spark-driver-memory",
        default="4g",
        help="Spark driver memory allocation (default: 4g).",
    )
    parser.add_argument(
        "--spark-executor-memory",
        default="8g",
        help="Spark executor memory allocation (default: 8g).",
    )
    args = parser.parse_args()
    config = OpenImagesConfig(
        data_dir=args.data_dir,
        detection_filename=args.annotation_file,
        class_filename=args.class_file,
        limit=args.limit,
        results_dir=args.results_dir,
        spark_master=args.spark_master,
        spark_driver_memory=args.spark_driver_memory,
        spark_executor_memory=args.spark_executor_memory,
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
            spark_master=base_config.spark_master,
            spark_driver_memory=base_config.spark_driver_memory,
            spark_executor_memory=base_config.spark_executor_memory,
        )
        configs.append((name, preset_config))
    return configs


def compute_size_bucket_shares(bucket_summary: pd.DataFrame) -> Dict[str, float]:
    if bucket_summary.empty:
        return {f"{bucket}_share": 0.0 for bucket in AREA_BUCKET_LABELS}

    bucket_counts = bucket_summary.set_index("size_bucket")["detections"].to_dict()
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
    summary_df = pd.DataFrame(records)
    summary_path = os.path.join(summary_dir, "openimages_spark_preset_summary.csv")
    summary_df.to_csv(summary_path, index=False)
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
