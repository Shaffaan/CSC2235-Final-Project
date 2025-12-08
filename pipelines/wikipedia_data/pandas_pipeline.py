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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    left_df: pd.DataFrame
    right_df: pd.DataFrame
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


def connect_to_db(config: Dict[str, Any]):
    """
    Opens a SQLite connection to the Wikipedia dataset.
    """
    dataset_root = config.get("dataset_root")
    if not dataset_root:
        dataset_root = ensure_sqlite_dataset(os.path.join(PROJECT_ROOT, "data", "sqlite_datasets"))
        config["dataset_root"] = dataset_root

    db_path = resolve_database_path(dataset_root, config.get("db_path"))
    config["db_path"] = db_path

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    print(f"  Connected to SQLite database at {db_path}")
    return connection


def load_tables(connection: sqlite3.Connection, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Loads the requested tables from SQLite into Pandas DataFrames.
    """
    table_limits = {**DEFAULT_TABLE_LIMITS, **config.get("table_limits", {})}
    tables_to_load: Sequence[str] = config.get("tables", list(table_limits.keys()))

    tables: Dict[str, pd.DataFrame] = {}
    for table_name in tables_to_load:
        limit = table_limits.get(table_name)
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        query = f"SELECT * FROM {table_name}{limit_clause}"

        start = time.perf_counter()
        df = pd.read_sql_query(query, connection)
        elapsed = time.perf_counter() - start

        tables[table_name] = df
        print(f"  Loaded {len(df):,} rows from '{table_name}' in {elapsed:.2f}s (limit={limit})")

    return tables


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def prepare_join_data(tables: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> List[JoinPlan]:
    """
    Builds JoinPlan definitions that describe how benchmarking should combine tables.
    """
    required_tables = {"pages", "items", "link_annotated_text"}
    missing = required_tables - tables.keys()
    if missing:
        raise KeyError(f"Missing tables required for join benchmarking: {', '.join(sorted(missing))}")

    pages = tables["pages"].copy()
    pages["item_id"] = _safe_numeric(pages["item_id"])

    items = tables["items"].copy()
    items["id"] = _safe_numeric(items["id"])
    items = items.rename(columns={"id": "item_id"})

    links = tables["link_annotated_text"].copy()

    properties = tables.get("properties")
    join_plans: List[JoinPlan] = [
        JoinPlan(
            name="pages_with_items",
            description="Enrich page metadata with Wikidata item labels.",
            left_label="pages",
            right_label="items",
            left_df=pages[["page_id", "item_id", "title", "views"]].dropna(subset=["item_id"]),
            right_df=items[["item_id", "labels", "description"]],
            left_on="item_id",
            right_on="item_id",
            suffixes=("_page", "_item"),
        ),
        JoinPlan(
            name="pages_with_links",
            description="Attach annotated link text to the core page dimension.",
            left_label="pages",
            right_label="link_annotated_text",
            left_df=pages[["page_id", "title", "views"]],
            right_df=links[["page_id", "sections"]],
            left_on="page_id",
            right_on="page_id",
            suffixes=("_page", "_link"),
        ),
    ]

    if properties is not None:
        props = properties.copy()
        props["id"] = _safe_numeric(props["id"])
        join_plans.append(
            JoinPlan(
                name="items_with_properties",
                description="Match overlapping identifiers between items and properties.",
                left_label="items",
                right_label="properties",
                left_df=items,
                right_df=props.rename(columns={"id": "item_id"}),
                left_on="item_id",
                right_on="item_id",
                suffixes=("_item", "_property"),
            )
        )

    print(f"  Prepared {len(join_plans)} join plans.")
    return join_plans


def benchmark_joins(join_plans: List[JoinPlan], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Executes a matrix of joins and records timing plus row counts.
    """
    join_types = config.get("join_types", DEFAULT_JOIN_TYPES)
    results_dir = config.get("results_dir") or os.environ.get("SCRIPT_RESULTS_DIR")
    if not results_dir:
        results_dir = os.path.join(PROJECT_ROOT, "results", "local_test", "wikipedia", "pandas")
    os.makedirs(results_dir, exist_ok=True)

    records: List[Dict[str, Any]] = []
    print(f"  Benchmarking join strategies ({', '.join(join_types)})...")

    for plan in join_plans:
        left_rows = len(plan.left_df)
        right_rows = len(plan.right_df)
        for join_type in join_types:
            step_label = f"{plan.name}:{join_type}"
            start = time.perf_counter()
            joined = pd.merge(
                plan.left_df,
                plan.right_df,
                how=join_type,
                left_on=plan.left_on,
                right_on=plan.right_on,
                suffixes=plan.suffixes,
            )
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
                    "result_rows": int(len(joined)),
                    "columns_returned": int(len(joined.columns)),
                    "execution_time_s": elapsed,
                }
            )
            print(f"    {step_label:<40} rows={len(joined):,} time={elapsed:.3f}s")
            del joined

    return pd.DataFrame.from_records(records)


def persist_join_metrics(results_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Saves the per-join benchmark metrics to disk and writes a JSON summary.
    """
    results_dir = config.get("results_dir") or os.environ.get("SCRIPT_RESULTS_DIR")
    if not results_dir:
        results_dir = os.path.join(PROJECT_ROOT, "results", "local_test", "wikipedia", "pandas")
    os.makedirs(results_dir, exist_ok=True)

    config_name = config.get("name", "unknown_config")
    csv_path = os.path.join(results_dir, f"join_metrics_{config_name}.csv")
    json_path = os.path.join(results_dir, f"join_summary_{config_name}.json")

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

    print(f"  Stored join metrics at {csv_path}")
    print(f"  Summary written to {json_path}")
    return results_df


def run_full_pipeline(config: Dict[str, Any], global_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the modular Wikipedia benchmark pipeline with per-step profiling.
    """
    print(f"\n[BEGIN] Running Wikipedia pipeline: {config['name']}...")
    pipeline_steps: List[Tuple[Any, bool]] = [
        (connect_to_db, False),
        (load_tables, True),
        (prepare_join_data, True),
        (benchmark_joins, True),
        (persist_join_metrics, True),
    ]

    process = psutil.Process(os.getpid())
    pipeline_start_time = time.perf_counter()
    artifact = None
    connection: Optional[sqlite3.Connection] = None
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
            if func is connect_to_db:
                connection = result

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
        print(f"!!! ERROR in pipeline config '{config['name']}' at step '{step_name}': {exc}", file=sys.stderr)
    finally:
        if connection:
            connection.close()
            print("  SQLite connection closed.")

    run_results_df = pd.DataFrame(run_results_list)
    combined = pd.concat([global_results_df, run_results_df], ignore_index=True)
    print(f"[END] Pipeline: {config['name']} finished.")
    return combined


if __name__ == "__main__":
    RESULTS_DIR = os.environ.get("SCRIPT_RESULTS_DIR")
    if not RESULTS_DIR:
        RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "local_test", "wikipedia", "pandas")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.environ["SCRIPT_RESULTS_DIR"] = RESULTS_DIR

    DATA_DOWNLOAD_ROOT = os.path.join(PROJECT_ROOT, "data", "sqlite_datasets")
    os.makedirs(DATA_DOWNLOAD_ROOT, exist_ok=True)

    SQLITE_DATASET_PATH = ensure_sqlite_dataset(DATA_DOWNLOAD_ROOT)

    print("Path to SQLite DB files:", SQLITE_DATASET_PATH)

    CONFIGURATIONS: List[Dict[str, Any]] = []
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
    for label, limits in size_options.items():
        CONFIGURATIONS.append(
            {
                "name": f"Wikipedia_{label}",
                "dataset_root": SQLITE_DATASET_PATH,
                "results_dir": RESULTS_DIR,
                "table_limits": limits,
                "join_types": DEFAULT_JOIN_TYPES,
                "sample_output_rows": 250,
            }
        )

    print(f"\n--- Generated {len(CONFIGURATIONS)} Wikipedia join benchmark configurations ---")
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    aggregate_csv = os.path.join(RESULTS_DIR, "wikipedia_pandas_join_pipeline.csv")
    all_results_df.to_csv(aggregate_csv, index=False)
    if all_results_df.empty:
        print("No step-level results were produced.")
    else:
        print(all_results_df[["config_name", "step", "execution_time_s", "peak_memory_mib"]])
    print(f"Full step metrics saved to {aggregate_csv}")
