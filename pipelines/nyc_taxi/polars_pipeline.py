import polars as pl
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

def calculate_and_save_stats(lf, config):
    """
    Calculates aggregate stats and saves them to a JSON file.
    Triggers the computation (.collect()) here.
    """
    print("  [ACTION] Executing full pipeline (collect)...")
    
    columns_to_check = [
        "trip_duration_mins", "speed_mph", "is_weekend",
        "fare_scaled", "dist_scaled", "speed_scaled"
    ]
    
    aggs = [pl.len().alias("total_rows")]
    for col in columns_to_check:
        aggs.append(pl.col(col).sum().alias(f"{col}_sum"))
        aggs.append(pl.col(col).mean().alias(f"{col}_avg"))
        aggs.append(pl.col(col).min().alias(f"{col}_min"))
        aggs.append(pl.col(col).max().alias(f"{col}_max"))
        aggs.append(pl.col(col).is_null().sum().alias(f"{col}_nulls"))

    # This is where the actual computation happens
    stats_df = lf.select(aggs).collect()
    
    stats_list = stats_df.to_dicts() 
    
    config_name = config.get('name', 'unknown_config')
    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config_name}.json')
    
    with open(output_path, 'w') as f:
        json.dump(stats_list, f, indent=2)
    
    print(f"  Final stats saved to {output_path}")
    return lf

def connect_to_db(config):
    """
    Sets global Polars thread pool size based on config.
    """
    num_threads = config.get('num_threads')
    if num_threads:
        os.environ["POLARS_MAX_THREADS"] = str(num_threads)
    
    # Enable streaming for large datasets to prevent OOM
    os.environ["POLARS_AUTO_STREAMING"] = "1"
    
    print(f"  Initialized Polars (Lazy). Threads: {os.environ.get('POLARS_MAX_THREADS')}, Streaming Enabled.")
    return None

def load_data(con, config):
    """
    Loads data using Polars' LAZY 'scan_parquet'.
    Returns a LazyFrame.
    """
    data_files = config['data_files']
    if not data_files:
        raise ValueError("No data files specified in config['data_files']")
    print(f"  Defining LazyFrame for {len(data_files)} files...")

    # union_by_name is handled efficiently by scan_parquet with wildcard or list
    lf = pl.scan_parquet(data_files)

    existing_cols = lf.columns

    schema_map = {
        'tpep_pickup_datetime':  ['tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime'],
        'tpep_dropoff_datetime': ['tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime'],
        'fare_amount':           ['fare_amount', 'Fare_Amt'],
        'trip_distance':         ['trip_distance', 'Trip_Distance'],
        'payment_type':          ['payment_type', 'Payment_Type']
    }

    select_expressions = []
    
    # We construct the schema unification expressions
    for canonical_name, source_options in schema_map.items():
        cols_that_exist = [col for col in source_options if col in existing_cols]
        
        if not cols_that_exist:
            select_expressions.append(pl.lit(None).alias(canonical_name))
        else:
            coalesce_exprs = [pl.col(c) for c in cols_that_exist]
            select_expressions.append(pl.coalesce(coalesce_exprs).alias(canonical_name))

    # Select only unified columns to prune the query plan early
    lf_unified = lf.select(select_expressions)

    print(f"  LazyFrame defined (schema unification added).")
    return lf_unified


def handle_missing_values(lf, config):
    print("  Adding imputation nodes to plan...")
    
    print("  [INTERMEDIATE] Computing global means for imputation...")
    means_df = lf.select([
        pl.col("fare_amount").filter(pl.col("fare_amount") > 0).mean().alias("mean_fare"),
        pl.col("trip_distance").filter(pl.col("trip_distance") > 0).mean().alias("mean_dist")
    ]).collect()
    
    mean_fare = means_df["mean_fare"][0] or 0.0
    mean_dist = means_df["mean_dist"][0] or 0.0
    
    lf_clean = lf.with_columns(
        pl.when(pl.col("fare_amount").is_null() | (pl.col("fare_amount") <= 0))
          .then(pl.lit(mean_fare))
          .otherwise(pl.col("fare_amount"))
          .alias("fare_amount_imputed"),
        
        pl.when(pl.col("trip_distance").is_null() | (pl.col("trip_distance") <= 0))
          .then(pl.lit(mean_dist))
          .otherwise(pl.col("trip_distance"))
          .alias("trip_distance_imputed"),
        
        pl.col("payment_type").cast(pl.Int64, strict=False).fill_null(0).alias("payment_type_imputed")
    )
    
    return lf_clean

def feature_engineering(lf, config):
    print("  Adding feature engineering nodes...")

    lf_with_duration = lf.with_columns(
        (
            (
                pl.col("tpep_dropoff_datetime").str.to_datetime(strict=False) -
                pl.col("tpep_pickup_datetime").str.to_datetime(strict=False)
            )
            .dt.total_seconds() / 60.0
        ).alias("trip_duration_mins")
    )

    lf_features = lf_with_duration.with_columns(
        pl.col("trip_duration_mins").fill_null(0).clip(lower_bound=0).alias("trip_duration_mins"),
        
        pl.col("tpep_pickup_datetime").str.to_datetime(strict=False)
          .dt.weekday().is_in([6, 7]).cast(pl.Int32).fill_null(0)
          .alias("is_weekend")
    ).with_columns(
        pl.when(pl.col("trip_duration_mins") > 0)
          .then(pl.col("trip_distance_imputed") / (pl.col("trip_duration_mins") / 60.0))
          .otherwise(0.0)
          .fill_null(0.0)
          .alias("speed_mph")
    )
    
    return lf_features

def categorical_encoding(lf, config):
    print("  [INTERMEDIATE] Scanning unique payment types...")
    # We must know unique values to build columns. 
    unique_df = lf.select("payment_type_imputed").unique().collect()
    payment_types = unique_df["payment_type_imputed"].to_list()
    
    print(f"  Found {len(payment_types)} payment types.")
    
    select_clauses = []
    for pt in payment_types:
        pt_str = str(pt) 
        pt_col_name = pt_str.replace('.', '_').replace('-', '_')
        
        clause = pl.when(pl.col("payment_type_imputed") == pt).then(1).otherwise(0).alias(f"payment_type_{pt_col_name}")
        select_clauses.append(clause)
        
    if not select_clauses:
        return lf
        
    lf_encoded = lf.with_columns(select_clauses)
    return lf_encoded

def numerical_scaling(lf, config):
    print("  [INTERMEDIATE] Aggregating Min/Max for Scaling...")
    # Calculate global min/max
    params_df = lf.select([
        pl.col("fare_amount_imputed").min().alias("min_fare"),
        pl.col("fare_amount_imputed").max().alias("max_fare"),
        pl.col("trip_distance_imputed").min().alias("min_dist"),
        pl.col("trip_distance_imputed").max().alias("max_dist"),
        pl.col("speed_mph").min().alias("min_speed"),
        pl.col("speed_mph").max().alias("max_speed"),
    ]).collect()

    params = params_df.row(0, named=True)

    min_fare = params.get("min_fare", 0) or 0
    max_fare = params.get("max_fare", 0) or 0
    min_dist = params.get("min_dist", 0) or 0
    max_dist = params.get("max_dist", 0) or 0
    min_speed = params.get("min_speed", 0) or 0
    max_speed = params.get("max_speed", 0) or 0

    print(f"  Scaling params: Fare({min_fare}, {max_fare}), Dist({min_dist}, {max_dist}), Speed({min_speed}, {max_speed})")

    def min_max_scaler(col_name, min_val, max_val):
        denominator = max_val - min_val
        if denominator == 0:
            return pl.lit(0.0)
        return (pl.col(col_name) - min_val) / denominator
        
    lf_scaled = lf.with_columns([
        min_max_scaler("fare_amount_imputed", min_fare, max_fare).alias("fare_scaled"),
        min_max_scaler("trip_distance_imputed", min_dist, max_dist).alias("dist_scaled"),
        min_max_scaler("speed_mph", min_speed, max_speed).alias("speed_scaled")
    ])
    
    return lf_scaled

def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']} (Polars LAZY)...")
    run_results_list = []
    pipeline_start_time = time.perf_counter()
    lf = None
    
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
                if lf is None and i == 1:
                    args = (None, config)
                else:
                    if lf is None:
                        print(f"  LazyFrame is missing. Stopping pipeline.")
                        break
                    args = (lf, config)

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
            
            lf = result
            
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
        import traceback
        traceback.print_exc()
    finally:
        del lf
        
    print(f"[END] Pipeline: {config['name']} finished.")
    run_results_df = pd.DataFrame(run_results_list)
    return pd.concat([global_results_df, run_results_df], ignore_index=True)


if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        print("Error: SCRIPT_RESULTS_DIR environment variable not set. Run this via run.sh")
        RESULTS_DIR = "results/local_test/nyc_taxi/polars"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    START_YEAR, START_MONTH = 2009, 1
    END_YEAR, END_MONTH = 2024, 12
    
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
        # '10%': all_files_sorted[:max(1, int(total_file_count * 0.10))],
        # '100%': all_files_sorted,
    }
    system_size_configs = {
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
    
    print(f"\n--- Generated {len(CONFIGURATIONS)} test configurations for Polars (Lazy) ---")
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, 'polars_results.csv')
    all_results_df.to_csv(output_csv, index=False)
    
    print(f"\n--- Polars benchmarks complete ---")
    if not all_results_df.empty:
        print(all_results_df[['config_name', 'step', 'execution_time_s', 'peak_memory_mib']])
    else:
        print("No results were generated.")
    print(f"Full results saved to {output_csv}")