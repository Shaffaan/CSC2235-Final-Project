import duckdb
import psutil
import pandas as pd
import time
import os
import sys
import glob
import urllib.request
import math
import warnings
from memory_profiler import memory_usage
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from common.download_utils import get_year_month_list, download_taxi_data

def calculate_and_save_stats(con, config):
    """
    Calculates aggregate stats and saves them to a JSON file.
    """
    print("  Calculating final statistics...")
    
    columns_to_check = [
        "trip_duration_mins", "speed_mph", "is_weekend",
        "fare_scaled", "dist_scaled", "speed_scaled"
    ]
    
    aggs = []
    for col in columns_to_check:
        aggs.append(f"SUM(CAST({col} AS DOUBLE)) AS {col}_sum")
        aggs.append(f"AVG(CAST({col} AS DOUBLE)) AS {col}_avg")
        aggs.append(f"MIN({col}) AS {col}_min")
        aggs.append(f"MAX({col}) AS {col}_max")
        aggs.append(f"COUNT(CASE WHEN {col} IS NULL THEN 1 END) AS {col}_nulls")

    query = f"""
    SELECT
        COUNT(*) AS total_rows,
        {', '.join(aggs)}
    FROM taxi_final
    """
    
    stats_df = con.execute(query).df()
    
    config_name = config.get('name', 'unknown_config')
    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config_name}.json')
    stats_df.to_json(output_path, orient="records", indent=2)
    
    print(f"  Final stats saved to {output_path}")
    return con

def connect_to_db(config):
    db_file = config.get('db_file', ':memory:')
    con = duckdb.connect(database=db_file, read_only=False)
    if config.get('memory_limit'):
        con.execute(f"SET memory_limit='{config['memory_limit']}'")
    if config.get('num_threads'):
        con.execute(f"SET threads={config['num_threads']}")
    print(f"  Connected to DB. Threads: {config.get('num_threads')}, Memory Limit: {config.get('memory_limit')}")
    return con

def load_data(con, config):
    data_files = config['data_files']
    if not data_files:
        raise ValueError("No data files specified in config['data_files']")
    print(f"  Loading and unifying schema for {len(data_files)} files...")

    con.execute(f"""
        CREATE TEMPORARY TABLE raw_data AS
        FROM read_parquet({data_files}, union_by_name=True)
    """)

    existing_cols = {r[1] for r in con.execute("PRAGMA table_info('raw_data')").fetchall()}

    schema_map = {
        'tpep_pickup_datetime':  ['tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime'],
        'tpep_dropoff_datetime': ['tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime'],
        'fare_amount':           ['fare_amount', 'Fare_Amt'],
        'trip_distance':         ['trip_distance', 'Trip_Distance'],
        'payment_type':          ['payment_type', 'Payment_Type']
    }

    select_clauses = []
    all_source_cols_to_exclude = set()

    for canonical_name, source_options in schema_map.items():
        cols_that_exist = [col for col in source_options if col in existing_cols]
        
        if not cols_that_exist:
            select_clauses.append(f"NULL AS {canonical_name}")
        else:
            quoted_cols = [
                f'"{c}"' if any(ch.isupper() for ch in c) else c 
                for c in cols_that_exist
            ]
            select_clauses.append(f"COALESCE({', '.join(quoted_cols)}) AS {canonical_name}")
            all_source_cols_to_exclude.update(cols_that_exist)

    final_exclude_list = sorted(existing_cols.intersection(all_source_cols_to_exclude))
    exclude_sql = ""
    if final_exclude_list:
        quoted_exclude_cols = [
            f'"{c}"' if any(ch.isupper() for ch in c) else c 
            for c in final_exclude_list
        ]
        exclude_sql = f"EXCLUDE ({', '.join(quoted_exclude_cols)})"

    con.execute(f"""
        CREATE TABLE taxi_data AS
        SELECT
            * {exclude_sql},
            {', '.join(select_clauses)}
        FROM raw_data
    """)

    con.execute("DROP TABLE raw_data")

    count = con.execute("SELECT COUNT(*) FROM taxi_data").fetchone()[0]
    print(f"  Loaded and unified {count} rows from {len(data_files)} files.")
    return con


def handle_missing_values(con, config):
    mean_fare_result = con.execute("SELECT AVG(fare_amount) FROM taxi_data WHERE fare_amount > 0").fetchone()
    mean_fare = mean_fare_result[0] if mean_fare_result and mean_fare_result[0] is not None else 0.0
    
    mean_distance_result = con.execute("SELECT AVG(trip_distance) FROM taxi_data WHERE trip_distance > 0").fetchone()
    mean_distance = mean_distance_result[0] if mean_distance_result and mean_distance_result[0] is not None else 0.0
    
    con.execute(f"""
    CREATE TABLE taxi_clean AS
    SELECT *,
        CASE WHEN fare_amount IS NULL OR fare_amount <= 0 THEN {mean_fare} ELSE fare_amount END AS fare_amount_imputed,
        CASE WHEN trip_distance IS NULL OR trip_distance <= 0 THEN {mean_distance} ELSE trip_distance END AS trip_distance_imputed,
        
        -- THE FIX: Cast payment_type to INTEGER and COALESCE NULLs to 0
        COALESCE(TRY_CAST(payment_type AS INTEGER), 0) AS payment_type_imputed
        
    FROM taxi_data
    """)
    print("  Imputed missing values.")
    return con

def feature_engineering(con, config):
    con.execute("ALTER TABLE taxi_clean ADD COLUMN trip_duration_mins DOUBLE; ALTER TABLE taxi_clean ADD COLUMN speed_mph DOUBLE; ALTER TABLE taxi_clean ADD COLUMN is_weekend INTEGER;")
    
    pickup_col = 'tpep_pickup_datetime'
    dropoff_col = 'tpep_dropoff_datetime'
    
    pickup_ts = f"TRY_CAST({pickup_col} AS TIMESTAMP)"
    dropoff_ts = f"TRY_CAST({dropoff_col} AS TIMESTAMP)"

    con.execute(f"""
        UPDATE taxi_clean 
        SET trip_duration_mins = CAST(
            epoch({dropoff_ts} - {pickup_ts}) 
        AS DOUBLE) / 60.0
    """)
    
    con.execute("UPDATE taxi_clean SET speed_mph = CASE WHEN trip_duration_mins > 0 THEN trip_distance_imputed / (trip_duration_mins / 60.0) ELSE 0 END;")
    
    con.execute(f"""
        UPDATE taxi_clean 
        SET is_weekend = CASE WHEN dayofweek({pickup_ts}) IN (0, 6) THEN 1 ELSE 0 END
    """)
    
    con.execute("UPDATE taxi_clean SET speed_mph = 0 WHERE speed_mph IS NULL OR isinf(speed_mph) OR isnan(speed_mph)")
    con.execute("UPDATE taxi_clean SET trip_duration_mins = 0 WHERE trip_duration_mins < 0 OR trip_duration_mins IS NULL")
    con.execute("UPDATE taxi_clean SET is_weekend = 0 WHERE is_weekend IS NULL")

    print("  Engineered features: duration, speed, is_weekend.")
    return con

def categorical_encoding(con, config):
    payment_types = [r[0] for r in con.execute("SELECT DISTINCT payment_type_imputed FROM taxi_clean").fetchall()]
    print(f"  One-hot encoding for payment types: {payment_types}")
    select_clauses = []
    for pt in payment_types:
        pt_str = str(pt) 
        pt_col_name = pt_str.replace('.', '_').replace('-', '_')
        
        pt_val = f"'{pt_str}'" if isinstance(pt, str) else pt 
        
        clause = f"CASE WHEN payment_type_imputed = {pt_val} THEN 1 ELSE 0 END AS payment_type_{pt_col_name}"
        select_clauses.append(clause)
        
    if not select_clauses:
        print("  No payment types found to encode. Skipping.")
        con.execute("CREATE TABLE taxi_encoded AS SELECT * FROM taxi_clean")
        return con
        
    con.execute(f"CREATE TABLE taxi_encoded AS SELECT *, {', '.join(select_clauses)} FROM taxi_clean")
    print("  Performed one-hot encoding.")
    return con

def numerical_scaling(con, config):
    con.execute("CREATE OR REPLACE MACRO min_max_scaler(val, min_val, max_val) AS (val - min_val) / nullif(max_val - min_val, 0)");
    params = con.execute("SELECT min(fare_amount_imputed), max(fare_amount_imputed), min(trip_distance_imputed), max(trip_distance_imputed), min(speed_mph), max(speed_mph) FROM taxi_encoded").fetchone()
    min_fare, max_fare, min_dist, max_dist, min_speed, max_speed = (val if val is not None else 0 for val in params)
    print(f"  Scaling params: Fare({min_fare}, {max_fare}), Dist({min_dist}, {max_dist}), Speed({min_speed}, {max_speed})")
    con.execute(f"""
    CREATE TABLE taxi_final AS
    SELECT *,
        min_max_scaler(fare_amount_imputed, {min_fare}, {max_fare}) AS fare_scaled,
        min_max_scaler(trip_distance_imputed, {min_dist}, {max_dist}) AS dist_scaled,
        min_max_scaler(speed_mph, {min_speed}, {max_speed}) AS speed_scaled
    FROM taxi_encoded
    """)
    print("  Applied Min-Max scaling.")
    return con

def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']}...")
    run_results_list = []
    pipeline_start_time = time.perf_counter()
    con = None
    pipeline_steps = [
        (connect_to_db, (config,), {}),
        (load_data, (None, config), {}),
        (handle_missing_values, (None, config), {}),
        (feature_engineering, (None, config), {}),
        (categorical_encoding, (None, config), {}),
        (numerical_scaling, (None, config), {}),
        (calculate_and_save_stats, (None, config), {})
    ]
    try:
        for i, (func, args, kwargs) in enumerate(pipeline_steps):
            step_name = func.__name__
            if i > 0:
                if con is None:
                    print(f"  Connection is missing. Stopping pipeline.")
                    break
                args = (con, config)
            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_times_before = process.cpu_times()
            step_start_time = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                mem_usage = memory_usage((func, args, kwargs), max_usage=True, retval=True, interval=0.1)
            peak_mem_mib = mem_usage[0]
            result = mem_usage[1]
            step_end_time = time.perf_counter()
            cpu_times_after = process.cpu_times()
            con = result
            elapsed_time = step_end_time - step_start_time
            cpu_time_used = (cpu_times_after.user - cpu_times_before.user) + (cpu_times_after.system - cpu_times_before.system)
            abs_start_time = step_start_time - pipeline_start_time
            abs_end_time = step_end_time - pipeline_start_time
            print(f"  [END] Step: {step_name}")
            print(f"    Exec Time: {elapsed_time:.4f} s")
            print(f"    CPU Time: {cpu_time_used:.4f} s")
            print(f"    Peak Mem: {peak_mem_mib:.2f} MiB")
            run_results_list.append({
                'config_name': config.get('name', 'N/A'),
                'step': step_name,
                'start_time': abs_start_time, 'end_time': abs_end_time,
                'execution_time_s': elapsed_time, 'cpu_time_s': cpu_time_used,
                'peak_memory_mib': peak_mem_mib, 'file_type': config.get('file_type', 'N/A'),
                'memory_limit': config.get('memory_limit', 'None'),
                'num_threads': config.get('num_threads', 'N/A'),
                'data_size_pct': config.get('data_size_pct', 'N/A'),
                'system_size_pct': config.get('system_size_pct', 'N/A')
            })
    except Exception as e:
        print(f"!!! ERROR in pipeline config '{config['name']}' at step '{step_name}': {e}", file=sys.stderr)
    finally:
        if con:
            try:
                con.close()
                print("  Connection closed.")
            except Exception as ce:
                print(f"  Error while closing connection: {ce}", file=sys.stderr)
    print(f"[END] Pipeline: {config['name']} finished.")
    run_results_df = pd.DataFrame(run_results_list)
    return pd.concat([global_results_df, run_results_df], ignore_index=True)

if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        print("Error: SCRIPT_RESULTS_DIR environment variable not set. Run this via run.sh")
        RESULTS_DIR = "results/local_test/nyc_taxi/duckdb"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    START_YEAR, START_MONTH = 2009, 1
    END_YEAR, END_MONTH = 2025, 9
    
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")
    
    year_month_list = get_year_month_list(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    
    all_files_sorted = download_taxi_data(year_month_list, LOCAL_DATA_DIR)
    
    if not all_files_sorted:
        print("No data files found or downloaded. Exiting benchmark.")
        sys.exit(1)

    try:
        total_mem_bytes = psutil.virtual_memory().total
        total_mem_gb = total_mem_bytes / (1024**3)
        total_cores = psutil.cpu_count(logical=True)
        print(f"System detected: {total_cores} logical cores, {total_mem_gb:.2f} GB RAM")
    except Exception:
        print("Could not detect system info, using defaults (4 cores, 8GB RAM).")
        total_mem_gb = 8.0
        total_cores = 4
    
    safe_max_mem_gb = total_mem_gb * 0.75
    total_file_count = len(all_files_sorted)
    
    data_size_configs = {
        '1%': all_files_sorted[:max(1, int(total_file_count * 0.01))],
        '2%': all_files_sorted[:max(1, int(total_file_count * 0.02))],
        # '30%': all_files_sorted[:max(1, int(total_file_count * 0.30))],
        # '100%': all_files_sorted,
    }
    system_size_configs = {
        # '10%': {'threads': max(1, int(total_cores * 0.10)), 'memory': f"{max(1, int(safe_max_mem_gb * 0.10))}GB"},
        # '30%': {'threads': max(1, int(total_cores * 0.30)), 'memory': f"{max(1, int(safe_max_mem_gb * 0.30))}GB"},
        '100%': {'threads': total_cores, 'memory': f"{int(safe_max_mem_gb)}GB"}
    }

    CONFIGURATIONS = []
    for data_pct, file_list in data_size_configs.items():
        for sys_pct, sys_config in system_size_configs.items():
            config_name = f"Data_{data_pct}_Sys_{sys_pct}"
            CONFIGURATIONS.append({
                "name": config_name, "file_type": "parquet", "data_files": file_list,
                "memory_limit": sys_config['memory'], "num_threads": sys_config['threads'],
                "data_size_pct": data_pct, "system_size_pct": sys_pct,
            })
    
    print(f"\n--- Generated {len(CONFIGURATIONS)} test configurations for DuckDB ---")
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, 'duckdb_results.csv')
    all_results_df.to_csv(output_csv, index=False)
    
    print(f"\n--- DuckDB benchmarks complete ---")
    if not all_results_df.empty:
        print(all_results_df[['config_name', 'step', 'execution_time_s', 'peak_memory_mib']])
    else:
        print("No results were generated.")
    print(f"Full results saved to {output_csv}")