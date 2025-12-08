import pandas as pd
import psutil
import time
import os
import sys
import warnings
from memory_profiler import memory_usage
import json

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from common.download_utils import download_aircheck_data

def calculate_and_save_stats(df, config):
    print("  [ACTION] Executing full pipeline (collect)...")
    columns_to_check = [
        "enrichment_score", "log_mw", "is_high_mw",
        "mw_scaled", "alogp_scaled", "enrichment_scaled"
    ]
    aggs = [F.count(F.lit(1)).alias("total_rows")]
    for col in columns_to_check:
        aggs.append(F.sum(col).alias(f"{col}_sum"))
        aggs.append(F.avg(col).alias(f"{col}_avg"))
        aggs.append(F.min(col).alias(f"{col}_min"))
        aggs.append(F.max(col).alias(f"{col}_max"))
        aggs.append(F.sum(F.when(F.col(col).isNull() | F.isnan(F.col(col)), 1).otherwise(0)).alias(f"{col}_nulls"))

    stats_row = df.agg(*aggs).collect()[0]
    stats_dict = stats_row.asDict()

    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config.get("name")}.json')
    with open(output_path, 'w') as f:
        json.dump([stats_dict], f, indent=2)
    print(f"  Final stats saved.")
    return df

def connect_to_db(config):
    mem = config.get('memory_limit', '8g')
    if "GB" in mem: mem = mem.replace("GB", "g")
    
    spark_mode = os.environ.get("SPARK_MODE", "local")
    master_url = os.environ.get("SPARK_MASTER_URL")
    
    builder = SparkSession.builder.appName(f"Aircheck_Benchmark_{spark_mode}")
    
    if spark_mode == "distributed" and master_url:
        print(f"  [Distributed] Connecting to Master: {master_url}")
        builder = builder.master(master_url)
        builder = builder.config("spark.executor.memory", mem)
        builder = builder.config("spark.driver.memory", "4g")
        builder = builder.config("spark.cores.max", int(config.get('num_threads', 12)) * 5)
    else:
        print(f"  [Local] Connecting to local[{config.get('num_threads', '*')}]")
        builder = builder.master(f"local[{config.get('num_threads', '*')}]")
        builder = builder.config("spark.driver.memory", mem)

    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    return session

def load_data(con, config):
    spark = con
    print(f"  Defining DataFrame (Parquet Read)...")
    cols = ['ALOGP', 'MW', 'NTC_VALUE', 'TARGET_VALUE', 'LABEL', 'BB1_ID', 'BB2_ID', 'TARGET_ID']
    
    files = [f"file://{os.path.abspath(f)}" for f in config['data_files']]
    
    df = spark.read.parquet(*files).select(*cols)
    
    df = df.withColumn("MW", F.col("MW").cast(DoubleType())) \
           .withColumn("ALOGP", F.col("ALOGP").cast(DoubleType())) \
           .withColumn("NTC_VALUE", F.col("NTC_VALUE").cast(DoubleType())) \
           .withColumn("TARGET_VALUE", F.col("TARGET_VALUE").cast(DoubleType()))
           
    return df

def handle_missing_values(df, config):
    print("  Adding imputation steps...")
    df = df.na.fill(0.0, subset=["TARGET_VALUE", "NTC_VALUE"]) \
           .na.fill("Unknown", subset=["BB1_ID", "BB2_ID", "TARGET_ID"])
    return df

def feature_engineering(df, config):
    print("  Adding feature engineering steps...")
    df = df.withColumn("enrichment_score", F.col("TARGET_VALUE") / (F.col("NTC_VALUE") + 1.0))
    
    df = df.withColumn("enrichment_score", 
         F.when(F.col("enrichment_score").isin([float('inf'), float('-inf')]), 0.0)
          .otherwise(F.col("enrichment_score")))

    df = df.withColumn("log_mw", F.log(F.col("MW") + 1.0)) \
            .withColumn("is_high_mw", F.when(F.col("MW") > 500, 1).otherwise(0))
            
    return df

def categorical_encoding(df, config):
    print("  [INTERMEDIATE] Aggregating Top 50 Targets...")
    
    top_rows = df.groupBy("TARGET_ID").count().orderBy(F.desc("count")).limit(50).collect()
    top_targets = [r['TARGET_ID'] for r in top_rows]
    
    print(f"  Found {len(top_targets)} targets.")
    
    exprs = []
    for t in top_targets:
        c_name = f"target_{str(t).replace('.', '_')}"
        exprs.append(F.when(F.col("TARGET_ID") == t, 1).otherwise(0).alias(c_name))
        
    df = df.select("*", *exprs)
    return df

def numerical_scaling(df, config):
    print("  [INTERMEDIATE] Aggregating Min/Max for Scaling...")
    
    stats = df.agg(
        F.min("MW").alias("min_mw"), F.max("MW").alias("max_mw"),
        F.min("ALOGP").alias("min_alogp"), F.max("ALOGP").alias("max_alogp"),
        F.min("enrichment_score").alias("min_en"), F.max("enrichment_score").alias("max_en")
    ).collect()[0]
    
    def scaler(col, min_v, max_v):
        if max_v is None or min_v is None or max_v == min_v: return F.lit(0.0)
        return (F.col(col) - min_v) / (max_v - min_v)
        
    df = df.withColumn("mw_scaled", scaler("MW", stats['min_mw'], stats['max_mw'])) \
            .withColumn("alogp_scaled", scaler("ALOGP", stats['min_alogp'], stats['max_alogp'])) \
            .withColumn("enrichment_scaled", scaler("enrichment_score", stats['min_en'], stats['max_en']))
            
    return df

def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']} (Spark Optimized)...")
    run_results_list = []
    pipeline_start_time = time.perf_counter()
    spark_session = None
    df = None
    
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
            if i == 0: current_args = (config,)
            elif i == 1: 
                if not spark_session: break
                current_args = (spark_session, config)
            else:
                if not df: break
                current_args = (df, config)
            
            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_before = process.cpu_times()
            start = time.perf_counter()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mem_usage = memory_usage((func, current_args, kwargs), max_usage=True, retval=True, interval=0.1)
            
            peak_mem = mem_usage[0]
            result = mem_usage[1]
            end = time.perf_counter()
            cpu_after = process.cpu_times()
            
            if i == 0: spark_session = result
            else: df = result
            
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
        if spark_session: spark_session.stop()
        
    return pd.concat([global_results_df, pd.DataFrame(run_results_list)], ignore_index=True)

if __name__ == "__main__":
    SPARK_MODE = os.environ.get("SPARK_MODE", "local")
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        RESULTS_DIR = f"results/local_test/aircheck/spark_{SPARK_MODE}"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "aircheck_wdr91")
    all_files = download_aircheck_data(LOCAL_DATA_DIR)
    
    if not all_files: sys.exit(1)

    try:
        total_mem_bytes = psutil.virtual_memory().total
        total_mem_gb = total_mem_bytes / (1024**3)
        total_threads = psutil.cpu_count(logical=True)
        print(f"System detected (Node): {total_threads} threads, {total_mem_gb:.2f} GB RAM")
    except Exception:
        print("Could not detect system info.")
    
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

    output_csv = os.path.join(RESULTS_DIR, f'spark_{SPARK_MODE}_results.csv')
    all_results_df.to_csv(output_csv, index=False)