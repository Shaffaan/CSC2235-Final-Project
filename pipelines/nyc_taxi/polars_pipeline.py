import polars as pl
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

def configure_polars(config):
    os.environ["POLARS_MAX_THREADS"] = str(config.get('num_threads', psutil.cpu_count(logical=True)))
    os.environ["POLARS_AUTO_STREAMING"] = "1"
    print(f"  [System] Polars Threads: {os.environ['POLARS_MAX_THREADS']}")

def execute_deterministic_pipeline(config):
    print("  Executing Deterministic Pipeline...")
    
    schema_overrides = {
        'Trip_Pickup_DateTime': pl.String,
        'Trip_Dropoff_DateTime': pl.String,
        'pickup_datetime': pl.String,
        'dropoff_datetime': pl.String,
        'Fare_Amt': pl.Float64,
        'Trip_Distance': pl.Float64,
        'Payment_Type': pl.Float64,
        
        'tpep_pickup_datetime': pl.Datetime("ns"),
        'tpep_dropoff_datetime': pl.Datetime("ns"),
        'fare_amount': pl.Float64,
        'trip_distance': pl.Float64,
        'payment_type': pl.Float64
    }

    lf = pl.scan_parquet(
        config['data_files'], 
        missing_columns="insert", 
        extra_columns="ignore",
        schema=schema_overrides 
    )
    
    date_cols = {
        'pickup_ts': {
            'str': ['Trip_Pickup_DateTime', 'pickup_datetime'],
            'dt': ['tpep_pickup_datetime']
        },
        'dropoff_ts': {
            'str': ['Trip_Dropoff_DateTime', 'dropoff_datetime'],
            'dt': ['tpep_dropoff_datetime']
        }
    }
    
    num_cols = {
        'fare_amt': ['fare_amount', 'Fare_Amt'],
        'dist_miles': ['trip_distance', 'Trip_Distance'],
        'payment_type': ['payment_type', 'Payment_Type']
    }
    
    sel = []
    
    for name, groups in date_cols.items():
        candidates = []
        for c in groups['dt']:
            candidates.append(pl.col(c).cast(pl.Datetime("us")))
        for c in groups['str']:
            candidates.append(pl.col(c).str.to_datetime(strict=False).cast(pl.Datetime("us")))
            
        sel.append(pl.coalesce(candidates).alias(name))

    for name, opts in num_cols.items():
        sel.append(pl.coalesce([pl.col(c) for c in opts]).alias(name))
    
    lf = lf.select(sel)

    lf = lf.filter(
        (pl.col("fare_amt") >= 1) & (pl.col("fare_amt") <= 500) &
        (pl.col("dist_miles") >= 0.1) & (pl.col("dist_miles") <= 100)
    )

    lf = lf.with_columns([
        ((pl.col("dropoff_ts") - pl.col("pickup_ts")).dt.total_seconds() / 60.0).alias("duration_mins"),
        (pl.col("pickup_ts").dt.weekday() >= 6).cast(pl.Int32).alias("is_weekend")
    ])

    lf = lf.filter((pl.col("duration_mins") >= 1) & (pl.col("duration_mins") <= 240))

    lf = lf.select([
        pl.col("duration_mins"),
        (pl.col("dist_miles") / (pl.col("duration_mins") / 60.0)).alias("speed_mph"),
        pl.col("is_weekend"),
        (pl.col("fare_amt") / 500.0).alias("fare_scaled"),
        (pl.col("dist_miles") / 100.0).alias("dist_scaled"),
    ])
    
    lf = lf.with_columns((pl.col("speed_mph") / 100.0).alias("speed_scaled"))
    
    cols = ["duration_mins", "speed_mph", "is_weekend", "fare_scaled", "dist_scaled", "speed_scaled"]
    aggs = [pl.len().alias("total_rows")] + [expr for c in cols for expr in [pl.col(c).mean().alias(f"{c}_avg"), pl.col(c).min().alias(f"{c}_min"), pl.col(c).max().alias(f"{c}_max")]]
    
    res = lf.select(aggs).collect(streaming=True)
    with open(os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config["name"]}.json'), 'w') as f:
        json.dump(res.to_dicts(), f, indent=2)
    
    print(f"  Rows Processed: {res['total_rows'][0]}")
    return res

def run_optimized_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Pipeline: {config['name']} (Polars)...")
    start = time.perf_counter()
    configure_polars(config)
    
    try:
        t0 = time.perf_counter(); c0 = psutil.Process().cpu_times()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem, _ = memory_usage((execute_deterministic_pipeline, (config,), {}), max_usage=True, retval=True, interval=0.1)
        t1 = time.perf_counter(); c1 = psutil.Process().cpu_times()

        stats = [{
            'config_name': config['name'], 'step': 'execute_deterministic_pipeline',
            'start_time': t0 - start, 'end_time': t1 - start,
            'execution_time_s': t1 - t0, 'cpu_time_s': (c1.user - c0.user) + (c1.system - c0.system),
            'peak_memory_mib': mem, 'file_type': 'parquet',
            'num_threads': config.get('num_threads'), 'data_size_pct': config.get('data_size_pct'),
            'system_size_pct': config.get('system_size_pct')
        }]
    except Exception as e:
        print(f"!!! Error: {e}"); import traceback; traceback.print_exc()
        stats = []

    return pd.concat([global_results_df, pd.DataFrame(stats)], ignore_index=True)

if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') or "results/local_test/nyc_taxi/polars"
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

    output_csv = os.path.join(RESULTS_DIR, 'polars_results.csv')
    all_results_df.to_csv(output_csv, index=False)