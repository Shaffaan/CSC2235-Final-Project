import json
import os
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import psutil
from memory_profiler import memory_usage
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from common.download_sqllite import ensure_sqlite_dataset

DEFAULT_TABLE_LIMITS = {
    "pages": 150_000,
    "items": 150_000,
    "link_annotated_text": 60_000,
    "properties": 50_000,
}
DEFAULT_JOIN_TYPES = ["inner", "left", "right", "outer"]


@dataclass
class JoinPlan:
    name: str
    description: str
    left_label: str
    right_label: str
    left_df: DataFrame
    right_df: DataFrame
    left_on: str
    right_on: str
    suffixes: Tuple[str, str]


def resolve_database_path(dataset_root: str, explicit_path: Optional[str] = None) -> str:
    """
    Attempts to resolve the SQLite DB path inside the downloaded dataset directory.
    """
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    known_path = os.path.join(dataset_root, "versions", "1", "kensho_dataset.db")
    if os.path.isfile(known_path):
        return known_path

    for root, _, files in os.walk(dataset_root):
        for file_name in files:
            if file_name.endswith(".db"):
                return os.path.join(root, file_name)
    raise FileNotFoundError(f"No SQLite DB file found inside {dataset_root}")


def create_spark_session(config: Dict[str, Any]) -> SparkSession:
    """
    Initializes a SparkSession for local or distributed execution.
    """
    spark_mode = os.environ.get("SPARK_MODE", "local")
    master_url = os.environ.get("SPARK_MASTER_URL")
    threads = config.get("num_threads", "*")
    exec_mem = config.get("memory_limit", "8g")

    builder = SparkSession.builder.appName(f"Wikipedia_Benchmark_{spark_mode}")

    if spark_mode == "distributed" and master_url:
        print(f"  [Distributed] Connecting Spark session to {master_url}")
        builder = builder.master(master_url)
        builder = builder.config("spark.executor.memory", exec_mem)
        builder = builder.config("spark.driver.memory", "4g")
        builder = builder.config("spark.cores.max", int(config.get("num_threads", 12)) * 5)
    else:
        print(f"  [Local] Starting Spark session with local[{threads}]")
        builder = builder.master(f"local[{threads}]")
        builder = builder.config("spark.driver.memory", exec_mem)

    builder = builder.config("spark.sql.shuffle.partitions", config.get("shuffle_partitions", 200))
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session


def load_tables(spark: SparkSession, config: Dict[str, Any]) -> Dict[str, DataFrame]:
    """
    Loads the requested SQLite tables into Spark DataFrames.
    """
    if spark is None:
        raise ValueError("Spark session not initialized.")

    dataset_root = config.get("dataset_root")
    if not dataset_root:
        dataset_root = ensure_sqlite_dataset(os.path.join(PROJECT_ROOT, "data", "sqlite_datasets"))
        config["dataset_root"] = dataset_root

    db_path = resolve_database_path(dataset_root, config.get("db_path"))
    config["db_path"] = db_path
    print(f"  Loading tables from SQLite DB at {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    table_limits = {**DEFAULT_TABLE_LIMITS, **config.get("table_limits", {})}
    tables_to_load: Sequence[str] = config.get("tables", list(table_limits.keys()))

    tables: Dict[str, DataFrame] = {}
    for table_name in tables_to_load:
        limit = table_limits.get(table_name)
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        query = f"SELECT * FROM {table_name}{limit_clause}"

        start = time.perf_counter()
        pdf = pd.read_sql_query(query, conn)
        elapsed = time.perf_counter() - start

        sdf = spark.createDataFrame(pdf)
        sdf = sdf.cache()
        row_count = sdf.count()

        tables[table_name] = sdf
        print(f"  Loaded {row_count:,} rows from '{table_name}' in {elapsed:.2f}s (limit={limit})")

        del pdf

    conn.close()
    return tables


def prepare_join_data(tables: Dict[str, DataFrame], config: Dict[str, Any]) -> List[JoinPlan]:
    """
    Builds JoinPlan definitions using Spark DataFrames mirroring the pandas pipeline.
    """
    required_tables = {"pages", "items", "link_annotated_text"}
    missing = required_tables - tables.keys()
    if missing:
        raise KeyError(f"Missing tables required for join benchmarking: {', '.join(sorted(missing))}")

    pages = tables["pages"].withColumn("item_id", F.col("item_id").cast("long"))
    items = tables["items"].withColumn("item_id", F.col("id").cast("long"))
    links = tables["link_annotated_text"]
    properties = tables.get("properties")

    page_dim = pages.select("page_id", "item_id", "title", "views").dropna(subset=["item_id"]).cache()
    item_dim = items.select("item_id", "labels", "description").cache()
    link_dim = links.select("page_id", "sections").cache()

    join_plans: List[JoinPlan] = [
        JoinPlan(
            name="pages_with_items",
            description="Enrich page metadata with Wikidata item labels.",
            left_label="pages",
            right_label="items",
            left_df=page_dim,
            right_df=item_dim,
            left_on="item_id",
            right_on="item_id",
            suffixes=("_page", "_item"),
        ),
        JoinPlan(
            name="pages_with_links",
            description="Attach annotated link text to the core page dimension.",
            left_label="pages",
            right_label="link_annotated_text",
            left_df=pages.select("page_id", "title", "views"),
            right_df=link_dim,
            left_on="page_id",
            right_on="page_id",
            suffixes=("_page", "_link"),
        ),
    ]

    if properties is not None:
        props = properties.withColumn("item_id", F.col("id").cast("long"))
        join_plans.append(
            JoinPlan(
                name="items_with_properties",
                description="Match overlapping identifiers between items and properties.",
                left_label="items",
                right_label="properties",
                left_df=items,
                right_df=props,
                left_on="item_id",
                right_on="item_id",
                suffixes=("_item", "_property"),
            )
        )

    print(f"  Prepared {len(join_plans)} Spark join plans.")
    return join_plans


def _rename_with_suffixes(df: DataFrame, rename_map: Dict[str, str]) -> DataFrame:
    """
    Helper to rename columns on a Spark DataFrame.
    """
    for source, target in rename_map.items():
        if source == target:
            continue
        df = df.withColumnRenamed(source, target)
    return df


def _prepare_join_inputs(plan: JoinPlan) -> Tuple[DataFrame, DataFrame]:
    """
    Applies suffixes to overlapping columns before performing joins (pandas-style).
    """
    left_suffix, right_suffix = plan.suffixes
    left_cols = set(plan.left_df.columns)
    right_cols = set(plan.right_df.columns)
    overlap = left_cols & right_cols

    left_renames: Dict[str, str] = {}
    right_renames: Dict[str, str] = {}
    for col in overlap:
        if col in (plan.left_on, plan.right_on):
            continue
        if left_suffix:
            left_renames[col] = f"{col}{left_suffix}"
        if right_suffix:
            right_renames[col] = f"{col}{right_suffix}"

    left_df = _rename_with_suffixes(plan.left_df, left_renames)
    right_df = _rename_with_suffixes(plan.right_df, right_renames)
    return left_df, right_df


def benchmark_joins(join_plans: List[JoinPlan], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Executes Spark joins for each JoinPlan/join type combination and records metrics.
    """
    join_types = config.get("join_types", DEFAULT_JOIN_TYPES)
    results_dir = config.get("results_dir") or os.environ.get("SCRIPT_RESULTS_DIR")
    if not results_dir:
        results_dir = os.path.join(PROJECT_ROOT, "results", "local_test", "wikipedia", "spark")
    os.makedirs(results_dir, exist_ok=True)
    config["results_dir"] = results_dir

    spark_join_map = {
        "inner": "inner",
        "left": "left_outer",
        "right": "right_outer",
        "outer": "full_outer",
    }

    records: List[Dict[str, Any]] = []
    print(f"  Benchmarking Spark joins ({', '.join(join_types)})...")

    for plan in join_plans:
        left_rows = plan.left_df.count()
        right_rows = plan.right_df.count()

        for join_type in join_types:
            spark_join_type = spark_join_map.get(join_type)
            if not spark_join_type:
                print(f"    Skipping unsupported join type '{join_type}'")
                continue

            left_df, right_df = _prepare_join_inputs(plan)
            condition = left_df[plan.left_on] == right_df[plan.right_on]

            step_label = f"{plan.name}:{join_type}"
            start = time.perf_counter()
            joined = left_df.join(right_df, condition, how=spark_join_type)
            result_rows = joined.count()
            elapsed = time.perf_counter() - start

            records.append(
                {
                    "config_name": config.get("name", "N/A"),
                    "join_plan": plan.name,
                    "description": plan.description,
                    "join_type": join_type,
                    "left_table": plan.left_label,
                    "right_table": plan.right_label,
                    "left_rows": int(left_rows),
                    "right_rows": int(right_rows),
                    "result_rows": int(result_rows),
                    "columns_returned": int(len(joined.columns)),
                    "execution_time_s": elapsed,
                }
            )
            print(f"    {step_label:<40} rows={result_rows:,} time={elapsed:.3f}s")

    return pd.DataFrame.from_records(records)


def persist_join_metrics(results_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Saves per-join benchmark metrics to disk and emits a JSON summary.
    """
    results_dir = config.get("results_dir") or os.environ.get("SCRIPT_RESULTS_DIR")
    if not results_dir:
        results_dir = os.path.join(PROJECT_ROOT, "results", "local_test", "wikipedia", "spark")
        os.makedirs(results_dir, exist_ok=True)

    config_name = config.get("name", "unknown_config")
    csv_path = os.path.join(results_dir, f"spark_join_metrics_{config_name}.csv")
    json_path = os.path.join(results_dir, f"spark_join_summary_{config_name}.json")

    results_df.to_csv(csv_path, index=False)

    summary: Dict[str, Any] = {
        "config_name": config_name,
        "total_runs": len(results_df),
        "fastest_join_s": float(results_df["execution_time_s"].min()) if not results_df.empty else None,
        "slowest_join_s": float(results_df["execution_time_s"].max()) if not results_df.empty else None,
    }
    if not results_df.empty:
        best_row = results_df.loc[results_df["execution_time_s"].idxmin()]
        summary["fastest_join_plan"] = f"{best_row['join_plan']} ({best_row['join_type']})"
        worst_row = results_df.loc[results_df["execution_time_s"].idxmax()]
        summary["slowest_join_plan"] = f"{worst_row['join_plan']} ({worst_row['join_type']})"

    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"  Stored Spark join metrics at {csv_path}")
    print(f"  Spark summary written to {json_path}")
    return results_df


def run_full_pipeline(config: Dict[str, Any], global_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the Spark Wikipedia pipeline with per-step profiling similar to pandas version.
    """
    print(f"\n[BEGIN] Running Wikipedia Spark pipeline: {config['name']}...")
    pipeline_steps: List[Tuple[Any, bool]] = [
        (create_spark_session, False),
        (load_tables, True),
        (prepare_join_data, True),
        (benchmark_joins, True),
        (persist_join_metrics, True),
    ]

    process = psutil.Process(os.getpid())
    pipeline_start_time = time.perf_counter()
    artifact: Any = None
    spark_session: Optional[SparkSession] = None
    run_results_list: List[Dict[str, Any]] = []

    step_name = "initialization"
    try:
        for func, needs_input in pipeline_steps:
            step_name = func.__name__
            args = (artifact, config) if needs_input else (config,)

            print(f"  [START] Step: {step_name}")
            cpu_times_before = process.cpu_times()
            step_start = time.perf_counter()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                mem_usage = memory_usage((func, args, {}), max_usage=True, retval=True, interval=0.1)

            peak_mem_mib = mem_usage[0]
            result = mem_usage[1]
            step_end = time.perf_counter()
            cpu_times_after = process.cpu_times()

            artifact = result
            if func is create_spark_session:
                spark_session = result

            elapsed = step_end - step_start
            cpu_time = (cpu_times_after.user - cpu_times_before.user) + (
                cpu_times_after.system - cpu_times_before.system
            )
            abs_start = step_start - pipeline_start_time
            abs_end = step_end - pipeline_start_time

            print(f"  [END] Step: {step_name}")
            print(f"    Exec Time: {elapsed:.4f} s")
            print(f"    CPU Time: {cpu_time:.4f} s")
            print(f"    Peak Mem: {peak_mem_mib:.2f} MiB")

            run_results_list.append(
                {
                    "config_name": config.get("name", "N/A"),
                    "step": step_name,
                    "start_time": abs_start,
                    "end_time": abs_end,
                    "execution_time_s": elapsed,
                    "cpu_time_s": cpu_time,
                    "peak_memory_mib": peak_mem_mib,
                    "table_limits": json.dumps(config.get("table_limits", {})),
                    "join_types": ",".join(config.get("join_types", DEFAULT_JOIN_TYPES)),
                }
            )
    except Exception as exc:
        print(f"!!! ERROR in Spark pipeline config '{config['name']}' at step '{step_name}': {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if spark_session:
            spark_session.stop()
            print("  Spark session stopped.")

    run_results_df = pd.DataFrame(run_results_list)
    combined = pd.concat([global_results_df, run_results_df], ignore_index=True)
    print(f"[END] Spark pipeline: {config['name']} finished.")
    return combined


if __name__ == "__main__":
    RESULTS_DIR = os.environ.get("SCRIPT_RESULTS_DIR")
    if not RESULTS_DIR:
        SPARK_MODE = os.environ.get("SPARK_MODE", "local")
        RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "local_test", "wikipedia", f"spark_{SPARK_MODE}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.environ["SCRIPT_RESULTS_DIR"] = RESULTS_DIR

    DATA_DOWNLOAD_ROOT = os.path.join(PROJECT_ROOT, "data", "sqlite_datasets")
    os.makedirs(DATA_DOWNLOAD_ROOT, exist_ok=True)

    SQLITE_DATASET_PATH = ensure_sqlite_dataset(DATA_DOWNLOAD_ROOT)

    print("Path to SQLite DB files:", SQLITE_DATASET_PATH)

    size_options = {
        "tiny": {"pages": 10_000, "items": 10_000, "link_annotated_text": 5_000, "properties": 3_000},
        "small": {"pages": 50_000, "items": 50_000, "link_annotated_text": 20_000, "properties": 10_000},
        # "medium": {"pages": 150_000, "items": 150_000, "link_annotated_text": 60_000, "properties": 25_000},
        # "large": {"pages": 300_000, "items": 300_000, "link_annotated_text": 120_000, "properties": 60_000},
        # "xlarge": {"pages": 600_000, "items": 600_000, "link_annotated_text": 240_000, "properties": 120_000},
        # "xxlarge": {"pages": 1_200_000, "items": 1_200_000, "link_annotated_text": 480_000, "properties": 200_000},
        # "mega": {"pages": 3_000_000, "items": 3_000_000, "link_annotated_text": 1_200_000, "properties": 400_000},
        # "full": {"pages": 5_362_174, "items": 6_824_000, "link_annotated_text": 5_343_565, "properties": 6_985},
    }

    CONFIGURATIONS: List[Dict[str, Any]] = []
    for label, limits in size_options.items():
        CONFIGURATIONS.append(
            {
                "name": f"Wikipedia_Spark_{label}",
                "dataset_root": SQLITE_DATASET_PATH,
                "results_dir": RESULTS_DIR,
                "table_limits": limits,
                "join_types": DEFAULT_JOIN_TYPES,
                "num_threads": psutil.cpu_count(logical=True) or "*",
                "memory_limit": "8g",
            }
        )

    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    aggregate_csv = os.path.join(RESULTS_DIR, "wikipedia_spark_join_pipeline.csv")
    all_results_df.to_csv(aggregate_csv, index=False)
    if all_results_df.empty:
        print("No step-level results were produced.")
    else:
        print(all_results_df[["config_name", "step", "execution_time_s", "peak_memory_mib"]])
    print(f"Full step metrics saved to {aggregate_csv}")
