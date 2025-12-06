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

from common.download_utils import download_aircheck_data

def calculate_and_save_stats(lf, config):
    print("  [ACTION] Executing full pipeline (collect)...")
    
    columns_to_check = [
        "enrichment_score", "log_mw", "is_high_mw",
        "mw_scaled", "alogp_scaled", "enrichment_scaled"
    ]
    aggs = [pl.count().alias("total_rows")]
    for col in columns_to_check:
        aggs.append(pl.col(col).sum().alias(f"{col}_sum"))
        aggs.append(pl.col(col).mean().alias(f"{col}_avg"))
        aggs.append(pl.col(col).min().alias(f"{col}_min"))
        aggs.append(pl.col(col).max().alias(f"{col}_max"))
        aggs.append(pl.col(col).is_null().sum().alias(f"{col}_nulls"))

    stats_df = lf.select(aggs).collect()
    
    stats_list = stats_df.to_dicts()
    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config.get("name")}.json')
    with open(output_path, 'w') as f:
        json.dump(stats_list, f, indent=2)
    print(f"  Final stats saved.")
    return lf

def connect_to_db(config):
    if config.get('num_threads'):
        os.environ["POLARS_MAX_THREADS"] = str(config.get('num_threads'))
    return None

def load_data(con, config):
    data_files = config['data_files']
    print(f"  Defining LazyFrame for {len(data_files)} file(s)...")
    
    cols_to_use = ['ALOGP', 'MW', 'NTC_VALUE', 'TARGET_VALUE', 'LABEL', 'BB1_ID', 'BB2_ID', 'TARGET_ID']
    
    lf = pl.scan_parquet(data_files).select(cols_to_use)
    
    lf = lf.with_columns([
        pl.col("MW").cast(pl.Float64),
        pl.col("ALOGP").cast(pl.Float64),
        pl.col("NTC_VALUE").cast(pl.Float64),
        pl.col("TARGET_VALUE").cast(pl.Float64)
    ])
    
    return lf

def handle_missing_values(lf, config):
    print("  Adding imputation nodes...")
    lf = lf.with_columns(
        pl.col("TARGET_VALUE").fill_null(0.0),
        pl.col("NTC_VALUE").fill_null(0.0),
        pl.col("BB1_ID").fill_null("Unknown"),
        pl.col("BB2_ID").fill_null("Unknown"),
        pl.col("TARGET_ID").fill_null("Unknown")
    )
    return lf

def feature_engineering(lf, config):
    print("  Adding feature engineering nodes...")
    
    lf = lf.with_columns(
        (pl.col("TARGET_VALUE") / (pl.col("NTC_VALUE") + 1.0)).alias("enrichment_score")
    )
    
    lf = lf.with_columns(
        pl.when(pl.col("enrichment_score").is_infinite()).then(0.0)
          .otherwise(pl.col("enrichment_score")).alias("enrichment_score")
    )

    lf = lf.with_columns(
        (pl.col("MW") + 1.0).log().alias("log_mw"),
        (pl.col("MW") > 500).cast(pl.Int32).alias("is_high_mw")
    )
    return lf

def categorical_encoding(lf, config):
    print("  [INTERMEDIATE] Aggregating Top 50 Targets...")
    
    top_targets_df = (
        lf.select("TARGET_ID")
          .group_by("TARGET_ID")
          .len()
          .sort("len", descending=True)
          .head(50)
          .collect()
    )
    top_targets = top_targets_df["TARGET_ID"].to_list()
    
    print(f"  Found {len(top_targets)} unique targets.")

    clauses = []
    for t in top_targets:
        t_clean = str(t).replace('.', '_')
        clauses.append(
            pl.when(pl.col("TARGET_ID") == t).then(1).otherwise(0).alias(f"target_{t_clean}")
        )
        
    if clauses:
        lf = lf.with_columns(clauses)
        
    return lf

def numerical_scaling(lf, config):
    print("  [INTERMEDIATE] Aggregating Min/Max for Scaling...")
    cols = ['MW', 'ALOGP', 'enrichment_score']
    aliases = ['mw_scaled', 'alogp_scaled', 'enrichment_scaled']
    
    stats_df = lf.select([
        pl.col(c).min().alias(f"{c}_min") for c in cols
    ] + [
        pl.col(c).max().alias(f"{c}_max") for c in cols
    ]).collect()
    
    stats = stats_df.row(0, named=True)
    
    scale_exprs = []
    for c, alias in zip(cols, aliases):
        min_v = stats[f"{c}_min"] or 0
        max_v = stats[f"{c}_max"] or 0
        denom = max_v - min_v
        if denom == 0:
            scale_exprs.append(pl.lit(0.0).alias(alias))
        else:
            scale_exprs.append(((pl.col(c) - min_v) / denom).alias(alias))
            
    lf = lf.with_columns(scale_exprs)
    return lf

def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']} (Polars Lazy)...")
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
                if lf is None and i == 1: args = (None, config)
                else:
                    if lf is None: break
                    args = (lf, config)
            
            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_before = process.cpu_times()
            start = time.perf_counter()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mem_usage = memory_usage((func, args, kwargs), max_usage=True, retval=True, interval=0.1)
            
            peak_mem = mem_usage[0]
            lf = mem_usage[1]
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
        del lf

    return pd.concat([global_results_df, pd.DataFrame(run_results_list)], ignore_index=True)

if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        RESULTS_DIR = "results/local_test/aircheck/polars"
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

    output_csv = os.path.join(RESULTS_DIR, 'polars_results.csv')
    all_results_df.to_csv(output_csv, index=False)