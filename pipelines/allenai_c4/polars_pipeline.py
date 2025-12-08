import polars as pl
import psutil
import pandas as pd
import time
import os
import sys
import warnings
from memory_profiler import memory_usage
import json
import hashlib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from common.download_utils import download_c4_data

def configure_polars(config):
    os.environ["POLARS_MAX_THREADS"] = str(config.get('num_threads', psutil.cpu_count(logical=True)))
    print(f"  [System] Polars Threads: {os.environ['POLARS_MAX_THREADS']}")

def execute_pipeline(config):
    print("  Executing C4 Pipeline (Polars Lazy)...")
    
    schema_overrides = {
        "url": pl.String,
        "text": pl.String,
        "timestamp": pl.String
    }
    
    lf = pl.scan_ndjson(config['data_files'], schema=schema_overrides)
    
    # 1. Filter
    lf = lf.filter(pl.col("text").str.len_chars() > 100)
    
    # 2. Transform (Extract and Lowercase)
    lf = lf.select([
        pl.col("url").str.extract(r"https?://([^/]+)", 1).str.to_lowercase().alias("domain"),
        pl.col("timestamp").str.slice(0, 4).alias("year")
    ])
    
    # Drop Nulls
    lf = lf.drop_nulls(subset=["domain"])
    
    # 3. Aggregate
    # Fix: Use .agg() to apply alias correctly
    aggs = lf.group_by(["domain", "year"]).agg(pl.len().alias("count")) \
             .sort(["count", "domain"], descending=[True, False]) \
             .limit(50)
    
    print("  Collecting results...")
    return aggs.collect()

def save_stats(df, config):
    print("  Saving stats...")
    raw_records = df.to_dicts()
    
    # --- SANITIZATION ---
    clean_records = []
    for r in raw_records:
        clean_records.append({
            "domain": str(r["domain"]).lower(),
            "year": str(r["year"]),
            "count": int(r["count"])
        })

    # Save Full Ranking
    ranking_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'ranking_{config["name"]}.json')
    with open(ranking_path, 'w') as f:
        json.dump(clean_records, f, indent=2)

    # Checksum
    checksum = hashlib.md5(json.dumps(clean_records, sort_keys=True).encode('utf-8')).hexdigest()
    
    stats_row = {
        "total_ranked": len(clean_records),
        "top_1_domain": clean_records[0]['domain'] if clean_records else "None",
        "top_1_count": clean_records[0]['count'] if clean_records else 0,
        "checksum": checksum
    }

    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config["name"]}.json')
    with open(output_path, 'w') as f:
        json.dump([stats_row], f, indent=2)
    return df

def run_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Pipeline: {config['name']}")
    start = time.perf_counter()
    configure_polars(config)
    
    try:
        t0 = time.perf_counter(); c0 = psutil.Process().cpu_times()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem, res_df = memory_usage((execute_pipeline, (config,), {}), max_usage=True, retval=True, interval=0.1)
        
        save_stats(res_df, config)
        
        t1 = time.perf_counter(); c1 = psutil.Process().cpu_times()

        stats = [{
            'config_name': config['name'], 'step': 'execute_pipeline',
            'start_time': t0 - start, 'end_time': t1 - start,
            'execution_time_s': t1 - t0, 'cpu_time_s': (c1.user - c0.user) + (c1.system - c0.system),
            'peak_memory_mib': mem, 'file_type': 'json.gz',
            'num_threads': config.get('num_threads'), 'data_size_pct': config.get('data_size_pct'),
            'system_size_pct': config.get('system_size_pct')
        }]
    except Exception as e:
        print(f"!!! Error: {e}"); import traceback; traceback.print_exc()
        stats = []

    return pd.concat([global_results_df, pd.DataFrame(stats)], ignore_index=True)

if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') or "results/local_test/allenai_c4/polars"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "allenai_c4")
    files = download_c4_data(LOCAL_DATA_DIR, num_files=100)
    
    config = {
        "name": "Data_100Files_Sys_100%", "data_files": files,
        "num_threads": psutil.cpu_count(logical=True),
        "data_size_pct": "100%", "system_size_pct": "100%"
    }
    
    run_pipeline(config, pd.DataFrame()).to_csv(os.path.join(RESULTS_DIR, 'polars_results.csv'), index=False)