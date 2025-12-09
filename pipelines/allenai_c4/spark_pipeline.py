import pandas as pd
import psutil
import time
import os
import sys
import json
import warnings
import hashlib
from memory_profiler import memory_usage

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from common.download_utils import download_c4_data

def connect_to_db(config):
    mem_limit = config.get('memory_limit', '8g').replace('GB', 'g')
    threads = config.get('num_threads', psutil.cpu_count(logical=True))
    
    spark_mode = os.environ.get("SPARK_MODE", "local")
    master_url = os.environ.get("SPARK_MASTER_URL")

    builder = SparkSession.builder \
        .appName(f"C4_{config['name']}") \
        .config("spark.driver.memory", mem_limit)

    if spark_mode == "distributed" and master_url:
        print(f"  [Distributed] Connecting to Master: {master_url}")
        builder = builder.master(master_url)
        builder = builder.config("spark.cores.max", int(threads) * 4) 
        builder = builder.config("spark.executor.memory", mem_limit)
    else:
        print(f"  [Local] Connecting to local[{threads}]")
        builder = builder.master(f"local[{threads}]")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def execute_pipeline(spark, config):
    print("  Executing C4 Pipeline (Spark)...")
    
    files = []
    for p in config['data_files']:
        if os.environ.get("SPARK_MODE") != "distributed" and not p.startswith("file://"):
            files.append(f"file://{os.path.abspath(p)}")
        else:
            files.append(p)

    schema = StructType([
        StructField("url", StringType(), True),
        StructField("text", StringType(), True),
        StructField("timestamp", StringType(), True)
    ])

    df = spark.read.schema(schema).json(files)
    
    # 1. Filter
    df = df.filter(F.length(F.col("text")) > 100)
    
    # 2. Transform (Extract and Lowercase)
    df = df.withColumn("domain", F.lower(F.regexp_extract(F.col("url"), r"https?://([^/]+)", 1)))
    df = df.withColumn("year", F.substring(F.col("timestamp"), 1, 4))
    
    # Drop empty strings (Regex failure)
    df = df.filter(F.col("domain") != "")
    
    # 3. Aggregate
    stats_df = df.groupBy("domain", "year").count() \
                 .orderBy(F.col("count").desc(), F.col("domain").asc()) \
                 .limit(50)
    
    print("  Collecting results...")
    return stats_df.collect()

def save_stats(rows, config):
    print("  Saving stats...")
    raw_records = [row.asDict() for row in rows]
    
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

def run_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Pipeline: {config['name']}")
    start = time.perf_counter()
    spark = None
    
    try:
        t0 = time.perf_counter(); c0 = psutil.Process().cpu_times()
        
        spark = connect_to_db(config)
        
        def _exec(): return execute_pipeline(spark, config)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem, rows = memory_usage((_exec, (), {}), max_usage=True, retval=True, interval=0.1)
        
        save_stats(rows, config)
        
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
    finally:
        if spark: spark.stop()

    return pd.concat([global_results_df, pd.DataFrame(stats)], ignore_index=True)

if __name__ == "__main__":
    SPARK_MODE = os.environ.get("SPARK_MODE", "local")
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        RESULTS_DIR = f"results/local_test/allenai_c4/spark_{SPARK_MODE}"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "allenai_c4")
    files = download_c4_data(LOCAL_DATA_DIR, num_files=100)
    
    try:
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        total_cores = psutil.cpu_count(logical=True)
    except:
        total_mem_gb = 8.0; total_cores = 4
    
    config = {
        "name": "Data_100Files_Sys_100%", "data_files": files,
        "num_threads": total_cores,
        "data_size_pct": "100%", "system_size_pct": "100%",
        "memory_limit": f"{int(total_mem_gb * 0.8)}GB"
    }
    
    run_pipeline(config, pd.DataFrame()).to_csv(os.path.join(RESULTS_DIR, f'spark_{SPARK_MODE}_results.csv'), index=False)