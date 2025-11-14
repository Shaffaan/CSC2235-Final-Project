import pandas as pd
import numpy as np
import psutil
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

def calculate_and_save_stats(df, config):
    """
    Calculates aggregate stats and saves them to a JSON file.
    """
    print("  Calculating final statistics...")
    
    columns_to_check = [
        "trip_duration_mins", "speed_mph", "is_weekend",
        "fare_scaled", "dist_scaled", "speed_scaled"
    ]
    
    stats = {"total_rows": int(len(df))}
    
    for col in columns_to_check:
        if col in df.columns:
            stats[f"{col}_sum"] = float(df[col].sum())
            stats[f"{col}_avg"] = float(df[col].mean())
            stats[f"{col}_min"] = float(df[col].min())
            stats[f"{col}_max"] = float(df[col].max())
            stats[f"{col}_nulls"] = int(df[col].isna().sum())
        else:
            print(f"  Warning: Column '{col}' not found in Pandas DataFrame. Skipping stats.")

    config_name = config.get('name', 'unknown_config')
    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config_name}.json')
    with open(output_path, 'w') as f:
        json.dump([stats], f, indent=2)
    
    print(f"  Final stats saved to {output_path}")
    return df

def connect_to_db(config):
    """
    A no-op function to fit the harness.
    Pandas doesn't have a 'connection' or global thread settings.
    """
    print(f"  Initialized Pandas. Threads: (Managed by NumPy/Pandas), Memory Limit: (System Managed)")
    return None

def load_data(con, config):
    """
    Loads data using Pandas.
    'con' (the connection) is ignored, as it's None.
    Returns an eager Pandas DataFrame.
    """
    data_files = config['data_files']
    if not data_files:
        raise ValueError("No data files specified in config['data_files']")
    print(f"  Loading and unifying schema for {len(data_files)} files...")

    df_list = (pd.read_parquet(f) for f in data_files)
    df = pd.concat(df_list, ignore_index=True)

    existing_cols = set(df.columns)

    schema_map = {
        'tpep_pickup_datetime':  ['tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime'],
        'tpep_dropoff_datetime': ['tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime'],
        'fare_amount':           ['fare_amount', 'Fare_Amt'],
        'trip_distance':         ['trip_distance', 'Trip_Distance'],
        'payment_type':          ['payment_type', 'Payment_Type']
    }

    new_cols_data = {}
    all_source_cols_to_drop = set()

    for canonical_name, source_options in schema_map.items():
        cols_that_exist = [col for col in source_options if col in existing_cols]
        
        if not cols_that_exist:
            new_cols_data[canonical_name] = pd.Series(np.nan, index=df.index)
        else:
            unified_col = df[cols_that_exist[0]]
            for other_col in cols_that_exist[1:]:
                unified_col = unified_col.fillna(df[other_col])
            
            new_cols_data[canonical_name] = unified_col
            all_source_cols_to_drop.update(cols_that_exist)

    cols_to_keep = [c for c in existing_cols if c not in all_source_cols_to_drop]
    
    df_unified = pd.concat(
        [df[cols_to_keep], pd.DataFrame(new_cols_data)], 
        axis=1
    )

    count = len(df_unified)
    print(f"  Loaded and unified {count} rows from {len(data_files)} files.")
    return df_unified


def handle_missing_values(df, config):
    """
    Accepts an eager DataFrame 'df' and returns a new eager DataFrame.
    """
    print("  Imputing missing values...")
    
    mean_fare = df.loc[df['fare_amount'] > 0, 'fare_amount'].mean()
    mean_distance = df.loc[df['trip_distance'] > 0, 'trip_distance'].mean()

    fare_condition = (df['fare_amount'].isna()) | (df['fare_amount'] <= 0)
    df['fare_amount_imputed'] = np.where(fare_condition, mean_fare, df['fare_amount'])

    dist_condition = (df['trip_distance'].isna()) | (df['trip_distance'] <= 0)
    df['trip_distance_imputed'] = np.where(dist_condition, mean_distance, df['trip_distance'])
    
    df['payment_type_imputed'] = pd.to_numeric(
        df['payment_type'], errors='coerce'
    ).fillna(0).astype(np.int64)
    
    print("  Imputed missing values.")
    return df

def feature_engineering(df, config):
    """
    Performs feature engineering using Pandas/Numpy.
    Accepts and returns an eager DataFrame.
    """
    print("  Engineering features...")

    pickup_ts = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff_ts = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    duration_delta = (dropoff_ts - pickup_ts)
    df['trip_duration_mins'] = duration_delta.dt.total_seconds() / 60.0
    
    df['is_weekend'] = pickup_ts.dt.dayofweek.isin([5, 6]).astype(int)

    df['trip_duration_mins'] = df['trip_duration_mins'].fillna(0).clip(lower=0)
    df['is_weekend'] = df['is_weekend'].fillna(0).astype(int)
    
    df['speed_mph'] = np.where(
        df['trip_duration_mins'] > 0, 
        df['trip_distance_imputed'] / (df['trip_duration_mins'] / 60.0), 
        0.0
    )
    df['speed_mph'] = df['speed_mph'].fillna(0.0)

    print("  Engineered features: duration, speed, is_weekend.")
    return df

def categorical_encoding(df, config):
    """
    Performs one-hot encoding on the 'payment_type_imputed' column.
    Accepts and returns an eager DataFrame.
    """
    payment_types = df["payment_type_imputed"].unique()
    print(f"  One-hot encoding for payment types: {payment_types}")
    
    dummies = pd.get_dummies(df['payment_type_imputed'], prefix='payment_type', dtype=int)
    
    df = pd.concat([df, dummies], axis=1)
        
    print("  Performed one-hot encoding.")
    return df

def numerical_scaling(df, config):
    """
    Applies Min-Max scaling.
    Accepts and returns an eager DataFrame.
    """
    
    cols_to_scale = {
        'fare_amount_imputed': 'fare_scaled',
        'trip_distance_imputed': 'dist_scaled',
        'speed_mph': 'speed_scaled'
    }
    
    scaling_params = {}

    for col_to_scale, new_col_name in cols_to_scale.items():
        min_val = df[col_to_scale].min()
        max_val = df[col_to_scale].max()
        
        scaling_params[col_to_scale] = (min_val, max_val)
        
        denominator = max_val - min_val
        if denominator == 0:
            df[new_col_name] = 0.0
        else:
            df[new_col_name] = (df[col_to_scale] - min_val) / denominator
    
    print(f"  Scaling params: {scaling_params}")
    return df

def run_full_pipeline(config, global_results_df):
    """
    Identical to the DuckDB/Polars harness.
    The 'con' variable will just hold a Pandas DataFrame
    which is passed from step to step.
    """
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
                if con is None and i == 1:
                    args = (None, config)
                else:
                    if con is None:
                        print(f"  DataFrame is missing. Stopping pipeline.")
                        break
                    args = (con, config)

            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_times_before = process.cpu_times()
            step_start_time = time.perf_counter()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
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
        del con
        print("  DataFrame cleared.")
        
    print(f"[END] Pipeline: {config['name']} finished.")
    run_results_df = pd.DataFrame(run_results_list)
    return pd.concat([global_results_df, run_results_df], ignore_index=True)


if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        print("Error: SCRIPT_RESULTS_DIR environment variable not set. Run this via run.sh")
        RESULTS_DIR = "results/local_test/nyc_taxi/pandas"
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
    
    print(f"\n--- Generated {len(CONFIGURATIONS)} test configurations for Pandas ---")
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, 'pandas_results.csv')
    all_results_df.to_csv(output_csv, index=False)
    
    print(f"\n--- Pandas benchmarks complete ---")
    if not all_results_df.empty:
        print(all_results_df[['config_name', 'step', 'execution_time_s', 'peak_memory_mib']])
    else:
        print("No results were generated.")
    print(f"Full results saved to {output_csv}")