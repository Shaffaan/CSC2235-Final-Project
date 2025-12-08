import duckdb
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

def connect_to_db(config):
    con = duckdb.connect(database=config.get('db_file', ':memory:'), read_only=False)
    if config.get('memory_limit'): con.execute(f"SET memory_limit='{config['memory_limit']}'")
    if config.get('num_threads'): con.execute(f"SET threads={config['num_threads']}")
    con.execute("PRAGMA enable_progress_bar")
    print(f"  Connected. Threads: {config.get('num_threads')}")
    return con

def execute_pipeline(con, config):
    print("  Executing C4 Pipeline (DuckDB)...")
    
    files = [f"'{f}'" for f in config['data_files']]
    files_str = ", ".join(files)

    # 1. Read JSON (Force types to avoid auto-detection issues)
    # 2. Filter length > 100
    # 3. Extract Domain (Lowercase) & Year
    # 4. Aggregate
    query = f"""
    CREATE OR REPLACE TABLE domain_stats AS
    WITH raw_data AS (
        SELECT * FROM read_json_auto([{files_str}], columns={{url: 'VARCHAR', text: 'VARCHAR', timestamp: 'VARCHAR'}})
    ),
    filtered AS (
        SELECT 
            url,
            timestamp
        FROM raw_data
        WHERE length(text) > 100
    ),
    transformed AS (
        SELECT
            lower(regexp_extract(url, 'https?://([^/]+)', 1)) as domain,
            substr(timestamp, 1, 4) as year
        FROM filtered
    )
    SELECT 
        domain,
        year,
        count(*) as count
    FROM transformed
    WHERE domain IS NOT NULL AND domain != ''
    GROUP BY domain, year
    ORDER BY count DESC, domain ASC
    LIMIT 50
    """
    
    con.execute(query)
    print(f"  Query Complete.")
    return con

def save_stats(con, config):
    print("  Saving stats...")
    df = con.execute("SELECT * FROM domain_stats").df()
    raw_records = df.to_dict(orient="records")

    # --- SANITIZATION FOR CHECKSUM PARITY ---
    # We enforce strict types (int, str lower) and keys to ensure checksums match across frameworks
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

    # Calculate Checksum
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
        
    print(f"  Checksum: {checksum}")
    return con

def run_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Pipeline: {config['name']}")
    start = time.perf_counter()
    con = None
    stats = []
    
    try:
        steps = [(connect_to_db, (config,), {}), (execute_pipeline, (None, config), {}), (save_stats, (None, config), {})]
        
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
                'peak_memory_mib': mem, 'file_type': 'json.gz',
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
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') or "results/local_test/allenai_c4/duckdb"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "allenai_c4")
    files = download_c4_data(LOCAL_DATA_DIR, num_files=100)
    
    config = {
        "name": "Data_100Files_Sys_100%", "data_files": files,
        "num_threads": psutil.cpu_count(logical=True), 
        "memory_limit": f"{int(psutil.virtual_memory().total / 1024**3 * 0.8)}GB",
        "data_size_pct": "100%", "system_size_pct": "100%"
    }
    
    run_pipeline(config, pd.DataFrame()).to_csv(os.path.join(RESULTS_DIR, 'duckdb_results.csv'), index=False)