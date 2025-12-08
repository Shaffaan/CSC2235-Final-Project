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

from common.download_utils import download_aircheck_data

def calculate_and_save_stats(rel, config):
    print("  [ACTION] Executing full pipeline (Calculation & Aggregation)...")
    
    columns_to_check = [
        "enrichment_score", "log_mw", "is_high_mw",
        "mw_scaled", "alogp_scaled", "enrichment_scaled"
    ]
    
    aggs = ["count(*) AS total_rows"]
    for col in columns_to_check:
        aggs.append(f"SUM(CAST({col} AS DOUBLE)) AS {col}_sum")
        aggs.append(f"AVG(CAST({col} AS DOUBLE)) AS {col}_avg")
        aggs.append(f"MIN({col}) AS {col}_min")
        aggs.append(f"MAX({col}) AS {col}_max")
        aggs.append(f"COUNT(CASE WHEN {col} IS NULL THEN 1 END) AS {col}_nulls")

    stats_df = rel.aggregate(", ".join(aggs)).df()
    
    config_name = config.get('name', 'unknown_config')
    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config_name}.json')
    stats_df.to_json(output_path, orient="records", indent=2)
    
    print(f"  Final stats saved to {output_path}")
    return rel

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
        raise ValueError("No data files specified")
    print(f"  Defining Lazy Relation for {len(data_files)} file(s)...")

    rel = con.read_parquet(data_files)
    
    rel = rel.project("""
        *,
        CAST(ALOGP AS DOUBLE) AS ALOGP_cast, 
        CAST(MW AS DOUBLE) AS MW_cast,
        CAST(TARGET_VALUE AS DOUBLE) AS TARGET_VALUE_cast,
        CAST(NTC_VALUE AS DOUBLE) AS NTC_VALUE_cast
    """)
    return rel

def handle_missing_values(rel, config):
    print("  Adding imputation nodes to query plan...")
    rel = rel.project("""
        *,
        COALESCE(TARGET_VALUE_cast, 0.0) AS TARGET_VALUE_imputed,
        COALESCE(NTC_VALUE_cast, 0.0) AS NTC_VALUE_imputed,
        COALESCE(BB1_ID, 'Unknown') AS BB1_ID_clean,
        COALESCE(BB2_ID, 'Unknown') AS BB2_ID_clean,
        COALESCE(TARGET_ID, 'Unknown') AS TARGET_ID_clean
    """)
    return rel

def feature_engineering(rel, config):
    print("  Adding feature engineering nodes to query plan...")
    
    rel = rel.project("""
        *,
        TARGET_VALUE_imputed / (NTC_VALUE_imputed + 1.0) AS enrichment_score_raw,
        ln(MW_cast + 1.0) AS log_mw,
        CASE WHEN MW_cast > 500 THEN 1 ELSE 0 END AS is_high_mw
    """)
    
    rel = rel.project("""
        *,
        CASE WHEN isinf(enrichment_score_raw) THEN 0.0 ELSE enrichment_score_raw END AS enrichment_score
    """)
    return rel

def categorical_encoding(rel, config):
    print("  [INTERMEDIATE] Aggregating Top 50 Targets...")
    
    top_targets_df = rel.aggregate("TARGET_ID_clean, count(*) AS cnt").order("cnt DESC").limit(50).df()
    target_ids = top_targets_df['TARGET_ID_clean'].tolist()
    
    print(f"  Found {len(target_ids)} unique targets. Adding encoding nodes...")

    case_statements = []
    for tid in target_ids:
        tid_safe = str(tid).replace('.', '_').replace('-', '_').replace(' ', '_')
        val_str = f"'{tid}'"
        statement = f"CASE WHEN TARGET_ID_clean = {val_str} THEN 1 ELSE 0 END AS target_{tid_safe}"
        case_statements.append(statement)
    
    if case_statements:
        rel = rel.project(f"*, {', '.join(case_statements)}")
        
    return rel

def numerical_scaling(rel, config):
    print("  [INTERMEDIATE] Aggregating Min/Max for Scaling...")
    
    stats = rel.aggregate("""
        MIN(MW_cast) as min_mw, MAX(MW_cast) as max_mw,
        MIN(ALOGP_cast) as min_alogp, MAX(ALOGP_cast) as max_alogp,
        MIN(enrichment_score) as min_enrich, MAX(enrichment_score) as max_enrich
    """).fetchone()
    
    min_mw, max_mw, min_alogp, max_alogp, min_enrich, max_enrich = (val if val is not None else 0 for val in stats)

    def get_scale_sql(col_name, min_v, max_v):
        denom = max_v - min_v
        if denom == 0: return "0.0"
        return f"({col_name} - {min_v}) / {denom}"

    mw_expr = get_scale_sql("MW_cast", min_mw, max_mw)
    alogp_expr = get_scale_sql("ALOGP_cast", min_alogp, max_alogp)
    enrich_expr = get_scale_sql("enrichment_score", min_enrich, max_enrich)

    rel = rel.project(f"""
        *,
        COALESCE({mw_expr}, 0) AS mw_scaled,
        COALESCE({alogp_expr}, 0) AS alogp_scaled,
        COALESCE({enrich_expr}, 0) AS enrichment_scaled
    """)
    
    return rel

def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']} (DuckDB)...")
    run_results_list = []
    pipeline_start_time = time.perf_counter()
    
    current_obj = None 
    con_ref = None

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
                if current_obj is None: break
                args = (current_obj, config)
            
            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_before = process.cpu_times()
            start = time.perf_counter()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mem_usage = memory_usage((func, args, kwargs), max_usage=True, retval=True, interval=0.1)
            
            peak_mem = mem_usage[0]
            current_obj = mem_usage[1]
            
            if i == 0: con_ref = current_obj
            
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
        print(f"!!! ERROR in pipeline {config['name']} step {step_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if con_ref: 
            try: con_ref.close()
            except: pass
    
    print(f"[END] Pipeline: {config['name']} finished.")
    return pd.concat([global_results_df, pd.DataFrame(run_results_list)], ignore_index=True)

if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        RESULTS_DIR = "results/local_test/aircheck/duckdb"
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
                "memory_limit": sys_config['memory'], "num_threads": sys_config['threads'],
                "data_size_pct": data_pct, "system_size_pct": sys_pct,
            })
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, 'duckdb_results.csv')
    all_results_df.to_csv(output_csv, index=False)