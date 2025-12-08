import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd


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
    # "large": 5_000_000,
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
        return os.path.join(PROJECT_ROOT, "results", "local_test", "cv", "openimages_duckdb")


def ensure_openimages_assets(config: OpenImagesConfig) -> Tuple[str, str]:
    annotations_path = download_openimages_detection(config.data_dir, config.detection_filename)
    class_map_path = download_openimages_class_descriptions(config.data_dir, config.class_filename)
    return annotations_path, class_map_path


def _quote_literal(value: str) -> str:
    return value.replace("'", "''")


def _table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> List[str]:
    return list(conn.execute(f"SELECT * FROM {table_name} LIMIT 0").fetchdf().columns)


def load_detections(
    conn: duckdb.DuckDBPyConnection, csv_path: str, limit: Optional[int]
) -> int:
    print(f"--- Loading detection annotations into DuckDB from {csv_path} ---")
    csv_literal = _quote_literal(csv_path)
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW detections_raw AS
        SELECT * FROM read_csv_auto('{csv_literal}', SAMPLE_SIZE=-1)
        """
    )
    if limit is not None:
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW detections AS SELECT * FROM detections_raw LIMIT {int(limit)}"
        )
    else:
        conn.execute("CREATE OR REPLACE TEMP VIEW detections AS SELECT * FROM detections_raw")
    row_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    print(f"  Loaded {row_count:,} detection rows (limit={limit})")
    return row_count


def load_class_descriptions(conn: duckdb.DuckDBPyConnection, csv_path: str) -> int:
    csv_literal = _quote_literal(csv_path)
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW class_map AS
        SELECT * FROM read_csv_auto('{csv_literal}', SAMPLE_SIZE=-1)
        """
    )
    columns = _table_columns(conn, "class_map")
    if not {"LabelName", "DisplayName"}.issubset(columns):
        raise KeyError("Expected columns 'LabelName' and 'DisplayName' in class descriptions")
    row_count = conn.execute("SELECT COUNT(*) FROM class_map").fetchone()[0]
    print(f"  Loaded {row_count:,} class description rows")
    return row_count


def compute_box_metrics(conn: duckdb.DuckDBPyConnection) -> None:
    required = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"}
    detection_columns = set(_table_columns(conn, "detections"))
    missing = required - detection_columns
    if missing:
        raise KeyError(f"Missing required detection columns: {', '.join(sorted(missing))}")

    has_image_dims = {"ImageWidth", "ImageHeight"}.issubset(detection_columns)
    image_area_expr = "CAST(ImageWidth AS DOUBLE) * CAST(ImageHeight AS DOUBLE)" if has_image_dims else "NULL"
    box_area_px_expr = (
        "box_width * CAST(ImageWidth AS DOUBLE) * box_height * CAST(ImageHeight AS DOUBLE)"
        if has_image_dims
        else "NULL"
    )
    normalized_expr = (
        f"""
        CASE
            WHEN {image_area_expr} IS NULL OR {image_area_expr} = 0 THEN box_area_ratio
            ELSE {box_area_px_expr} / NULLIF({image_area_expr}, 0)
        END
        """
        if has_image_dims
        else "box_area_ratio"
    )

    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW detection_metrics AS
        WITH base AS (
            SELECT
                ImageID,
                LabelName AS class_id,
                CAST(XMin AS DOUBLE) AS XMin,
                CAST(XMax AS DOUBLE) AS XMax,
                CAST(YMin AS DOUBLE) AS YMin,
                CAST(YMax AS DOUBLE) AS YMax,
                GREATEST(CAST(XMax AS DOUBLE) - CAST(XMin AS DOUBLE), 0.0) AS box_width,
                GREATEST(CAST(YMax AS DOUBLE) - CAST(YMin AS DOUBLE), 0.0) AS box_height,
                GREATEST(
                    GREATEST(CAST(XMax AS DOUBLE) - CAST(XMin AS DOUBLE), 0.0)
                    * GREATEST(CAST(YMax AS DOUBLE) - CAST(YMin AS DOUBLE), 0.0),
                    0.0
                ) AS box_area_ratio,
                {image_area_expr} AS image_area_px_raw,
                {box_area_px_expr} AS box_area_px_raw,
                {normalized_expr} AS normalized_area_raw
            FROM detections
        ),
        normalized AS (
            SELECT
                *,
                CASE
                    WHEN normalized_area_raw IS NULL THEN 0.0
                    ELSE LEAST(GREATEST(normalized_area_raw, 0.0), 1.0)
                END AS normalized_area,
                image_area_px_raw AS image_area_px,
                box_area_px_raw AS box_area_px
            FROM base
        )
        SELECT
            *,
            CASE
                WHEN normalized_area < {AREA_BUCKET_EDGES[1]} THEN '{AREA_BUCKET_LABELS[0]}'
                WHEN normalized_area < {AREA_BUCKET_EDGES[2]} THEN '{AREA_BUCKET_LABELS[1]}'
                ELSE '{AREA_BUCKET_LABELS[2]}'
            END AS size_bucket
        FROM normalized
        """
    )


def attach_class_names(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TEMP VIEW detections_enriched AS
        SELECT
            metrics.*,
            class_map.DisplayName AS class_name
        FROM detection_metrics AS metrics
        LEFT JOIN class_map
            ON metrics.class_id = class_map.LabelName
        """
    )


def summarize_class_areas(conn: duckdb.DuckDBPyConnection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    area_stats = conn.execute(
        """
        SELECT
            class_id,
            class_name,
            COUNT(*) AS detections,
            AVG(normalized_area) AS mean_norm_area,
            MEDIAN(normalized_area) AS median_norm_area,
            STDDEV_POP(normalized_area) AS std_norm_area
        FROM detections_enriched
        GROUP BY class_id, class_name
        ORDER BY detections DESC
        """
    ).fetchdf()

    bucket_counts = conn.execute(
        """
        SELECT
            class_id,
            class_name,
            size_bucket,
            COUNT(*) AS detections
        FROM detections_enriched
        GROUP BY class_id, class_name, size_bucket
        ORDER BY class_id, size_bucket
        """
    ).fetchdf()

    return area_stats, bucket_counts


def summarize_bucket_distribution(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            size_bucket,
            COUNT(*) AS detections,
            AVG(normalized_area) AS avg_normalized_area
        FROM detections_enriched
        GROUP BY size_bucket
        ORDER BY size_bucket
        """
    ).fetchdf()


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
    conn = duckdb.connect(database=":memory:")
    try:
        start = time.perf_counter()
        rows_loaded = load_detections(conn, annotations_path, config.limit)
        _record_timing(timings, "load_detections", start, rows=rows_loaded)

        start = time.perf_counter()
        classes_loaded = load_class_descriptions(conn, class_map_path)
        _record_timing(timings, "load_class_map", start, classes=classes_loaded)

        start = time.perf_counter()
        compute_box_metrics(conn)
        attach_class_names(conn)
        enriched_rows = conn.execute("SELECT COUNT(*) FROM detections_enriched").fetchone()[0]
        _record_timing(timings, "compute_metrics_and_join", start, rows=enriched_rows)

        start = time.perf_counter()
        area_stats, bucket_counts = summarize_class_areas(conn)
        _record_timing(
            timings,
            "group_area_by_class",
            start,
            classes=len(area_stats),
            bucket_rows=len(bucket_counts),
        )

        start = time.perf_counter()
        bucket_summary = summarize_bucket_distribution(conn)
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
        conn.close()


def parse_args() -> Tuple[OpenImagesConfig, Optional[str]]:
    parser = argparse.ArgumentParser(
        description="Open Images bounding box statistics with DuckDB"
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
        help="Directory to store CSV summaries (defaults to results/local_test/cv/openimages_duckdb).",
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
    summary_path = os.path.join(summary_dir, "openimages_duckdb_preset_summary.csv")
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
