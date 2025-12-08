import pandas as pd
import numpy as np
import psutil
import time
import os
import sys
import warnings
from memory_profiler import memory_usage
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from common.download_utils import get_year_month_list, download_taxi_data

def configure_pandas(config):
    threads = config.get('num_threads', psutil.cpu_count(logical=True))
    print(f"  [System] Pandas detected threads: {threads}")

def load_and_standardize_chunk(file_path):
    """
    Reads a single Parquet file, standardizes column names/types, 
    and applies 'Pre-Filtering' to reduce memory usage before concatenation.
    """
    schema_map = {
        'pickup_ts': ['tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime'],
        'dropoff_ts': ['tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime'],
        'fare_amt': ['fare_amount', 'Fare_Amt'],
        'dist_miles': ['trip_distance', 'Trip_Distance'],
        'payment_type': ['payment_type', 'Payment_Type']
    }
    
    all_possible_cols = [col for cols in schema_map.values() for col in cols]
    
    try:
        import pyarrow.parquet as pq
        file_schema = pq.read_schema(file_path).names
        cols_to_read = [c for c in all_possible_cols if c in file_schema]
        
        if not cols_to_read:
            return None

        df = pd.read_parquet(file_path, columns=cols_to_read)
        
        for canonical, options in schema_map.items():
            found = [c for c in options if c in df.columns]
            if found:
                df.rename(columns={found[0]: canonical}, inplace=True)
                if len(found) > 1:
                    df.drop(columns=found[1:], inplace=True)
            else:
                df[canonical] = np.nan

        df['pickup_ts'] = pd.to_datetime(df['pickup_ts'], errors='coerce')
        df['dropoff_ts'] = pd.to_datetime(df['dropoff_ts'], errors='coerce')
        
        num_cols = ['fare_amt', 'dist_miles', 'payment_type']
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

        mask = (
            (df['fare_amt'] >= 1) & (df['fare_amt'] <= 500) &
            (df['dist_miles'] >= 0.1) & (df['dist_miles'] <= 100)
        )
        return df.loc[mask].copy()

    except Exception as e:
        print(f"    Warning: Failed to process {os.path.basename(file_path)}: {e}")
        return None

def execute_deterministic_pipeline(config):
    print("  Executing Deterministic Pipeline...")
    
    chunk_list = []
    for f in config['data_files']:
        chunk = load_and_standardize_chunk(f)
        if chunk is not None and not chunk.empty:
            chunk_list.append(chunk)
            
    if not chunk_list:
        raise ValueError("No data loaded after filtering!")

    print(f"  Concatenating {len(chunk_list)} file chunks...")
    df = pd.concat(chunk_list, ignore_index=True)
    del chunk_list

    df['duration_mins'] = (df['dropoff_ts'] - df['pickup_ts']).dt.total_seconds() / 60.0
    
    df['is_weekend'] = df['pickup_ts'].dt.dayofweek.isin([5, 6]).astype('int32')

    mask_duration = (df['duration_mins'] >= 1) & (df['duration_mins'] <= 240)
    df = df.loc[mask_duration].copy()

    df['speed_mph'] = df['dist_miles'] / (df['duration_mins'] / 60.0)
    
    df['fare_scaled'] = df['fare_amt'] / 500.0
    df['dist_scaled'] = df['dist_miles'] / 100.0
    df['speed_scaled'] = df['speed_mph'] / 100.0

    final_cols = ["duration_mins", "speed_mph", "is_weekend", "fare_scaled", "dist_scaled", "speed_scaled"]
    df = df[final_cols].astype('float32')
    
    print(f"  Rows Processed: {len(df)}")
    return df

def calculate_stats(df, config):
    print("  Calculating stats...")
    stats = {
        "total_rows": int(len(df))
    }
    
    cols = ["duration_mins", "speed_mph", "is_weekend", "fare_scaled", "dist_scaled", "speed_scaled"]
    
    for col in cols:
        if col in df.columns:
            stats[f"{col}_avg"] = float(df[col].mean())
            stats[f"{col}_min"] = float(df[col].min())
            stats[f"{col}_max"] = float(df[col].max())

    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config["name"]}.json')
    with open(output_path, 'w') as f:
        json.dump([stats], f, indent=2)
    
    return df

def run_optimized_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Pipeline: {config['name']}")
    start = time.perf_counter()
    configure_pandas(config)
    
    try:
        t0 = time.perf_counter(); c0 = psutil.Process().cpu_times()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem, df_res = memory_usage((execute_deterministic_pipeline, (config,), {}), max_usage=True, retval=True, interval=0.1)
        
        calculate_stats(df_res, config)
        
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
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') or "results/local_test/nyc_taxi/pandas"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")
    files = download_taxi_data(get_year_month_list(2009, 1, 2024, 12), LOCAL_DATA_DIR)
    
    config = {
        "name": "Data_20%_Sys_100%", "data_files": files[:max(1, int(len(files) * 0.20))],
        "num_threads": psutil.cpu_count(logical=True),
        "data_size_pct": "20%", "system_size_pct": "100%"
    }
    
    run_optimized_pipeline(config, pd.DataFrame()).to_csv(os.path.join(RESULTS_DIR, 'pandas_results.csv'), index=False)