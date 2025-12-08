import duckdb
import psutil
import pandas as pd
import time
import os
import sys
import warnings
from memory_profiler import memory_usage
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from common.download_utils import get_year_month_list, download_taxi_data

def connect_to_db(config):
    con = duckdb.connect(database=config.get('db_file', ':memory:'), read_only=False)
    if config.get('memory_limit'): con.execute(f"SET memory_limit='{config['memory_limit']}'")
    if config.get('num_threads'): con.execute(f"SET threads={config['num_threads']}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA enable_progress_bar")
    print(f"  Connected. Threads: {config.get('num_threads')}")
    return con

def execute_deterministic_pipeline(con, config):
    print("  Executing Deterministic Pipeline...")
    con.execute(f"CREATE OR REPLACE VIEW raw_view AS FROM read_parquet({config['data_files']}, union_by_name=True)")
    
    existing = {r[1] for r in con.execute("PRAGMA table_info('raw_view')").fetchall()}
    
    schema = {
        'pickup': (['tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime'], 'VARCHAR'),
        'dropoff': (['tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime'], 'VARCHAR'),
        'fare': (['fare_amount', 'Fare_Amt'], 'DOUBLE'),
        'dist': (['trip_distance', 'Trip_Distance'], 'DOUBLE'),
        'ptype': (['payment_type', 'Payment_Type'], 'VARCHAR')
    }
    
    cols = {}
    for name, (opts, target_type) in schema.items():
        found = [c for c in opts if c in existing]
        if found:
            coalesce_args = [f"CAST(\"{c}\" AS {target_type})" if any(x.isupper() for x in c) else f"CAST({c} AS {target_type})" for c in found]
            cols[name] = f"COALESCE({', '.join(coalesce_args)})"
        else:
            cols[name] = "NULL"

    query = f"""
    CREATE TABLE taxi_final AS
    WITH clean_inputs AS (
        SELECT 
            TRY_CAST({cols['pickup']} AS TIMESTAMP) as pickup_ts,
            TRY_CAST({cols['dropoff']} AS TIMESTAMP) as dropoff_ts,
            TRY_CAST({cols['fare']} AS DOUBLE) as fare_amt,
            TRY_CAST({cols['dist']} AS DOUBLE) as dist_miles,
            TRY_CAST({cols['ptype']} AS INTEGER) as payment_type
        FROM raw_view
    ),
    filtered AS (
        SELECT * FROM clean_inputs
        WHERE fare_amt BETWEEN 1 AND 500
          AND dist_miles BETWEEN 0.1 AND 100
    ),
    features AS (
        SELECT *,
            date_diff('second', pickup_ts, dropoff_ts) / 60.0 as duration_mins,
            CASE WHEN dayofweek(pickup_ts) IN (0, 6) THEN 1 ELSE 0 END as is_weekend
        FROM filtered
    ),
    final_calc AS (
        SELECT *,
            CASE WHEN duration_mins >= 1 THEN dist_miles / (duration_mins / 60.0) ELSE 0 END as speed_mph
        FROM features
        WHERE duration_mins BETWEEN 1 AND 240
    )
    SELECT
        duration_mins, speed_mph, is_weekend,
        (fare_amt - 0) / 500.0 as fare_scaled,
        (dist_miles - 0) / 100.0 as dist_scaled,
        (speed_mph - 0) / 100.0 as speed_scaled
    FROM final_calc
    """
    con.execute(query)
    print(f"  Rows Processed: {con.execute('SELECT COUNT(*) FROM taxi_final').fetchone()[0]}")
    return con

def calculate_stats(con, config):
    print("  Calculating stats...")
    cols = ["duration_mins", "speed_mph", "is_weekend", "fare_scaled", "dist_scaled", "speed_scaled"]
    aggs = [f"AVG({c}) AS {c}_avg, MIN({c}) AS {c}_min, MAX({c}) AS {c}_max" for c in cols]
    df = con.execute(f"SELECT COUNT(*) AS total_rows, {', '.join(aggs)} FROM taxi_final").df()
    df.to_json(os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config["name"]}.json'), orient="records", indent=2)
    return con

def run_optimized_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Pipeline: {config['name']} (DuckDB)...")
    start = time.perf_counter()
    con = None
    stats = []
    
    try:
        steps = [(connect_to_db, (config,), {}), (execute_deterministic_pipeline, (None, config), {}), (calculate_stats, (None, config), {})]
        
        for func, args, kwargs in steps:
            if con: args = (con, config)
            t0 = time.perf_counter(); c0 = psutil.Process().cpu_times()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mem, res = memory_usage((func, args, kwargs), max_usage=True, retval=True, interval=0.1)
            con = res
            t1 = time.perf_counter(); c1 = psutil.Process().cpu_times()
            stats.append({
                'config_name': config['name'], 'step': func.__name__,
                'start_time': t0 - start, 'end_time': t1 - start,
                'execution_time_s': t1 - t0, 'cpu_time_s': (c1.user - c0.user) + (c1.system - c0.system),
                'peak_memory_mib': mem, 'file_type': 'parquet',
                'num_threads': config.get('num_threads'), 'data_size_pct': config.get('data_size_pct'),
                'system_size_pct': config.get('system_size_pct')
            })
            print(f"  [Step] {func.__name__} ({t1-t0:.2f}s)")

    except Exception as e:
        print(f"!!! Error: {e}"); import traceback; traceback.print_exc()
    finally:
        if con: con.close()

    return pd.concat([global_results_df, pd.DataFrame(stats)], ignore_index=True)

if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') or "results/local_test/nyc_taxi/duckdb"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")
    files = download_taxi_data(get_year_month_list(2009, 1, 2024, 12), LOCAL_DATA_DIR)
    
    if not files: sys.exit(1)

    try:
        total_mem_bytes = psutil.virtual_memory().total
        total_mem_gb = total_mem_bytes / (1024**3)
        total_threads = psutil.cpu_count(logical=True)
        print(f"System detected: {total_threads} threads, {total_mem_gb:.2f} GB RAM")
    except Exception:
        print("Could not detect system info.")
        total_threads = 4
        total_mem_gb = 8

    total_file_count = len(files)
    
    data_size_configs = {
        '100%': files
    }
    
    system_size_configs = {
        '12t_16gb': {'threads': 12, 'memory': "16GB"},
        '20t_64gb': {'threads': 20, 'memory': "64GB"},
        '40t_128gb': {'threads': 40, 'memory': "128GB"},
    }

    CONFIGURATIONS = []
    for data_pct, file_list in data_size_configs.items():
        for sys_pct, sys_config in system_size_configs.items():
            if total_threads < sys_config['threads'] or total_mem_gb < int(sys_config['memory'][:-2]):
                print(f"  Skipping config {data_pct} data / {sys_pct} system due to insufficient resources.")
                continue
            
            config_name = f"Data_{data_pct}_Sys_{sys_pct}"
            CONFIGURATIONS.append({
                "name": config_name,
                "file_type": "parquet",
                "data_files": file_list,
                "memory_limit": sys_config['memory'],
                "num_threads": sys_config['threads'],
                "data_size_pct": data_pct,
                "system_size_pct": sys_pct,
            })
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_optimized_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, 'duckdb_results.csv')
    all_results_df.to_csv(output_csv, index=False)