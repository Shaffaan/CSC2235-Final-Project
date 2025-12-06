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

from common.download_utils import download_aircheck_data

def calculate_and_save_stats(df, config):
    print("  Calculating final statistics...")
    columns_to_check = [
        "enrichment_score", "log_mw", "is_high_mw",
        "mw_scaled", "alogp_scaled", "enrichment_scaled"
    ]
    stats = {"total_rows": int(len(df))}
    for col in columns_to_check:
        if col in df.columns:
            stats[f"{col}_sum"] = float(df[col].sum())
            stats[f"{col}_avg"] = float(df[col].mean())
            stats[f"{col}_min"] = float(df[col].min())
            stats[f"{col}_max"] = float(df[col].max())
            stats[f"{col}_nulls"] = int(df[col].isna().sum())

    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config.get("name")}.json')
    with open(output_path, 'w') as f:
        json.dump([stats], f, indent=2)
    print(f"  Final stats saved to {output_path}")
    return df

def connect_to_db(config):
    return None

def load_data(con, config):
    data_files = config['data_files']
    print(f"  Loading {len(data_files)} file(s)...")
    
    cols_to_use = ['ALOGP', 'MW', 'NTC_VALUE', 'TARGET_VALUE', 'LABEL', 'BB1_ID', 'BB2_ID', 'TARGET_ID']
    
    df_list = []
    for f in data_files:
        df_list.append(pd.read_parquet(f, columns=cols_to_use))
        
    df = pd.concat(df_list, ignore_index=True)
    
    df['MW'] = df['MW'].astype(np.float64)
    df['ALOGP'] = df['ALOGP'].astype(np.float64)
    df['NTC_VALUE'] = df['NTC_VALUE'].astype(np.float64)
    df['TARGET_VALUE'] = df['TARGET_VALUE'].astype(np.float64)
    
    print(f"  Loaded {len(df)} rows.")
    return df

def handle_missing_values(df, config):
    print("  Imputing missing values...")
    df['TARGET_VALUE'] = df['TARGET_VALUE'].fillna(0.0)
    df['NTC_VALUE'] = df['NTC_VALUE'].fillna(0.0)
    
    for col in ['BB1_ID', 'BB2_ID', 'TARGET_ID']:
        df[col] = df[col].fillna('Unknown')
        
    return df

def feature_engineering(df, config):
    print("  Engineering features...")
    
    denominator = df['NTC_VALUE'] + 1.0
    df['enrichment_score'] = df['TARGET_VALUE'] / denominator
    
    df['enrichment_score'] = df['enrichment_score'].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df['log_mw'] = np.log(df['MW'].clip(lower=0) + 1.0)
    
    df['is_high_mw'] = (df['MW'] > 500).astype(int)
    
    return df

def categorical_encoding(df, config):
    print("  One-hot encoding TARGET_ID...")
    top_targets = df['TARGET_ID'].value_counts().nlargest(50).index
    
    mask = df['TARGET_ID'].isin(top_targets)
    
    temp_series = df['TARGET_ID'].where(mask, 'Other')
    dummies = pd.get_dummies(temp_series, prefix='target', dtype=int)
    
    if 'target_Other' in dummies.columns:
        dummies.drop(columns=['target_Other'], inplace=True)
        
    df = pd.concat([df, dummies], axis=1)
    return df

def numerical_scaling(df, config):
    print("  Applying Min-Max scaling...")
    cols_to_scale = {
        'MW': 'mw_scaled',
        'ALOGP': 'alogp_scaled',
        'enrichment_score': 'enrichment_scaled'
    }
    for col, new_col in cols_to_scale.items():
        min_val = df[col].min()
        max_val = df[col].max()
        
        if pd.isna(min_val) or pd.isna(max_val):
             df[new_col] = 0.0
             continue

        denom = max_val - min_val
        if denom == 0:
            df[new_col] = 0.0
        else:
            df[new_col] = (df[col] - min_val) / denom
            
    return df

def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']} (Pandas)...")
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
                else:
                    if con is None: break
                    args = (con, config)
            
            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_before = process.cpu_times()
            start = time.perf_counter()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mem_usage = memory_usage((func, args, kwargs), max_usage=True, retval=True, interval=0.1)
            
            peak_mem = mem_usage[0]
            con = mem_usage[1]
            end = time.perf_counter()
            cpu_after = process.cpu_times()
            
            elapsed = end - start
            cpu_used = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
            
            run_results_list.append({
                'config_name': config.get('name'), 'step': step_name,
                'start_time': start - pipeline_start_time, 'end_time': end - pipeline_start_time,
                'execution_time_s': elapsed, 'cpu_time_s': cpu_used, 'peak_memory_mib': peak_mem,
                'system_size_pct': config.get('system_size_pct'), 'data_size_pct': config.get('data_size_pct')
            })
            print(f"  [END] Step: {step_name} ({elapsed:.4f}s)")
    except Exception as e:
        print(f"!!! ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        del con

    return pd.concat([global_results_df, pd.DataFrame(run_results_list)], ignore_index=True)

if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        RESULTS_DIR = "results/local_test/aircheck/pandas"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aircheck_wdr91")
    all_files = download_aircheck_data(LOCAL_DATA_DIR)
    
    if not all_files: sys.exit(1)

    try:
        total_mem_bytes = psutil.virtual_memory().total
        total_mem_gb = total_mem_bytes / (1024**3)
        total_threads = psutil.cpu_count(logical=True)
        print(f"System detected: {total_threads} threads, {total_mem_gb:.2f} GB RAM")
    except Exception:
        print("Could not detect system info.")
    
    total_file_count = len(all_files)
    
    data_size_configs = {
        '100%': all_files,
    }
    system_size_configs = {
        '12t_16gb': {'threads': 12, 'memory': "16GB"}
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
                "memory_limit": sys_config['memory'], "num_threads": sys_config['threads'],
                "data_size_pct": data_pct, "system_size_pct": sys_pct,
            })
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, 'pandas_results.csv')
    all_results_df.to_csv(output_csv, index=False)