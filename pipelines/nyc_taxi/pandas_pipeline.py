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
import gc
from memory_profiler import memory_usage
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from common.download_utils import get_year_month_list, download_taxi_data

def calculate_and_save_stats(df, config):
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

    config_name = config.get('name', 'unknown_config')
    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config_name}.json')
    with open(output_path, 'w') as f:
        json.dump([stats], f, indent=2)
    
    print(f"  Final stats saved to {output_path}")
    return df

def connect_to_db(config):
    return None

def load_data(con, config):
    data_files = config['data_files']
    if not data_files:
        raise ValueError("No data files specified")
    print(f"  Loading and unifying schema for {len(data_files)} files...")

    df_list = []
    for f in data_files:
        df_temp = pd.read_parquet(f)
        fcols = df_temp.select_dtypes('float').columns
        df_temp[fcols] = df_temp[fcols].astype('float32')
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()

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
            new_cols_data[canonical_name] = pd.Series(np.nan, index=df.index, dtype='float32')
        else:
            unified_col = df[cols_that_exist[0]]
            for other_col in cols_that_exist[1:]:
                unified_col = unified_col.fillna(df[other_col])
            new_cols_data[canonical_name] = unified_col
            all_source_cols_to_drop.update(cols_that_exist)

    cols_to_keep = [c for c in existing_cols if c not in all_source_cols_to_drop]
    df_unified = pd.concat([df[cols_to_keep], pd.DataFrame(new_cols_data)], axis=1)
    
    del df
    gc.collect()
    
    print(f"  Loaded {len(df_unified)} rows.")
    return df_unified

def handle_missing_values(df, config):
    print("  Imputing missing values...")
    
    mean_fare = df.loc[df['fare_amount'] > 0, 'fare_amount'].mean()
    mean_distance = df.loc[df['trip_distance'] > 0, 'trip_distance'].mean()

    df['fare_amount_imputed'] = df['fare_amount'].fillna(mean_fare)
    mask_fare = df['fare_amount_imputed'] <= 0
    df.loc[mask_fare, 'fare_amount_imputed'] = mean_fare
    
    df['trip_distance_imputed'] = df['trip_distance'].fillna(mean_distance)
    mask_dist = df['trip_distance_imputed'] <= 0
    df.loc[mask_dist, 'trip_distance_imputed'] = mean_distance

    df['payment_type_imputed'] = pd.to_numeric(df['payment_type'], errors='coerce').fillna(0).astype('int32')
    
    return df

def feature_engineering(df, config):
    print("  Engineering features...")
    
    pickup_ts = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff_ts = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    df['trip_duration_mins'] = ((dropoff_ts - pickup_ts).dt.total_seconds() / 60.0).astype('float32')
    
    # Clean duration
    df['trip_duration_mins'] = df['trip_duration_mins'].fillna(0).clip(lower=0)
    
    df['is_weekend'] = pickup_ts.dt.dayofweek.isin([5, 6]).astype('int32')
    
    # Calculate speed
    df['speed_mph'] = (df['trip_distance_imputed'] / (df['trip_duration_mins'] / 60.0)).replace([np.inf, -np.inf], 0).fillna(0).astype('float32')

    del pickup_ts, dropoff_ts
    gc.collect()
    return df

def categorical_encoding(df, config):
    dummies = pd.get_dummies(df['payment_type_imputed'], prefix='payment_type', dtype='int8')
    df = pd.concat([df, dummies], axis=1)
    return df

def numerical_scaling(df, config):
    print("  Applying Min-Max scaling...")
    cols = {
        'fare_amount_imputed': 'fare_scaled',
        'trip_distance_imputed': 'dist_scaled',
        'speed_mph': 'speed_scaled'
    }
    
    for col, new_col in cols.items():
        min_v = df[col].min()
        max_v = df[col].max()
        denom = max_v - min_v
        if denom == 0:
            df[new_col] = 0.0
        else:
            df[new_col] = ((df[col] - min_v) / denom).astype('float32')
            
    return df

def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']} (Pandas Optimized)...")
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
                if con is None and i == 1: args = (None, config)
                else: args = (con, config)

            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_times_before = process.cpu_times()
            step_start_time = time.perf_counter()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mem_usage = memory_usage((func, args, kwargs), max_usage=True, retval=True, interval=0.1)
            
            peak_mem_mib = mem_usage[0]
            con = mem_usage[1]
            step_end_time = time.perf_counter()
            cpu_times_after = process.cpu_times()
            
            elapsed_time = step_end_time - step_start_time
            cpu_time_used = (cpu_times_after.user - cpu_times_before.user) + (cpu_times_after.system - cpu_times_before.system)
            
            print(f"  [END] Step: {step_name} ({elapsed_time:.4f}s)")
            
            run_results_list.append({
                'config_name': config.get('name'), 'step': step_name,
                'start_time': step_start_time - pipeline_start_time, 'end_time': step_end_time - pipeline_start_time,
                'execution_time_s': elapsed_time, 'cpu_time_s': cpu_time_used, 'peak_memory_mib': peak_mem_mib,
                'system_size_pct': config.get('system_size_pct'), 'data_size_pct': config.get('data_size_pct')
            })
    except Exception as e:
        print(f"!!! ERROR: {e}")
    finally:
        del con
        gc.collect()

    return pd.concat([global_results_df, pd.DataFrame(run_results_list)], ignore_index=True)

if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        RESULTS_DIR = "results/local_test/nyc_taxi/pandas"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    START_YEAR, START_MONTH = 2009, 1
    END_YEAR, END_MONTH = 2024, 12
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")
    year_month_list = get_year_month_list(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    all_files_sorted = download_taxi_data(year_month_list, LOCAL_DATA_DIR)
    
    if not all_files_sorted: sys.exit(1)

    try:
        total_mem_bytes = psutil.virtual_memory().total
        total_mem_gb = total_mem_bytes / (1024**3)
        total_cores = psutil.cpu_count(logical=True)
        print(f"System detected: {total_cores} threads, {total_mem_gb:.2f} GB RAM")
    except Exception:
        total_mem_gb = 8.0
        total_cores = 4
    
    safe_max_mem_gb = total_mem_gb * 0.75
    total_file_count = len(all_files_sorted)
    
    data_size_configs = {
        '1%': all_files_sorted[:max(1, int(total_file_count * 0.01))],
        # '10%': all_files_sorted[:max(1, int(total_file_count * 0.10))],
    }
    system_size_configs = {
        '100%': {'threads': total_cores, 'memory': f"{int(safe_max_mem_gb)}GB"}
    }

    CONFIGURATIONS = []
    for data_pct, file_list in data_size_configs.items():
        for sys_pct, sys_config in system_size_configs.items():
            CONFIGURATIONS.append({
                "name": f"Data_{data_pct}_Sys_{sys_pct}", "file_type": "parquet", "data_files": file_list,
                "memory_limit": sys_config['memory'], "num_threads": sys_config['threads'],
                "data_size_pct": data_pct, "system_size_pct": sys_pct,
            })
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    all_results_df.to_csv(os.path.join(RESULTS_DIR, 'pandas_results.csv'), index=False)