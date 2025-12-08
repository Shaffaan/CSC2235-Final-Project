import pandas as pd
import psutil
import time
import os
import sys
import json
import warnings
from functools import reduce
from memory_profiler import memory_usage

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from common.download_utils import get_year_month_list, download_taxi_data

def connect_to_db(config):
    mem_limit = config.get('memory_limit', '8g').replace('GB', 'g')
    threads = config.get('num_threads', psutil.cpu_count(logical=True))
    
    spark_mode = os.environ.get("SPARK_MODE", "local")
    master_url = os.environ.get("SPARK_MASTER_URL")

    builder = SparkSession.builder \
        .appName(f"NYC_Taxi_{config['name']}") \
        .config("spark.driver.memory", mem_limit) \
        .config("spark.sql.parquet.mergeSchema", "false") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.ui.showConsoleProgress", "true")

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

def standardize_chunk(df):
    """
    Renames columns to a standard schema and casts types strictly.
    This runs lazily on every file before Union.
    """
    cols = set(df.columns)
    
    def get_col(candidates):
        for c in candidates:
            if c in cols: return F.col(c)
        return F.lit(None)

    sel = [
        get_col(['tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime']).cast(StringType()).alias("pickup_ts"),
        get_col(['tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime']).cast(StringType()).alias("dropoff_ts"),
        get_col(['fare_amount', 'Fare_Amt']).cast(DoubleType()).alias("fare_amt"),
        get_col(['trip_distance', 'Trip_Distance']).cast(DoubleType()).alias("dist_miles"),
        get_col(['payment_type', 'Payment_Type']).cast(DoubleType()).alias("payment_type")
    ]
    
    return df.select(*sel)

def execute_deterministic_pipeline(spark, config):
    print("  Executing Deterministic Pipeline...")
    
    dfs = []
    
    for p in config['data_files']:
        path = f"file://{os.path.abspath(p)}" if os.environ.get("SPARK_MODE") != "distributed" else p
        df_chunk = spark.read.option("mergeSchema", "false").parquet(path)
        dfs.append(standardize_chunk(df_chunk))
        
    if not dfs:
        raise ValueError("No data files loaded.")

    df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)

    df = df.withColumn("pickup_ts", F.to_timestamp("pickup_ts")) \
           .withColumn("dropoff_ts", F.to_timestamp("dropoff_ts"))

    df = df.filter(
        (F.col("fare_amt") >= 1) & (F.col("fare_amt") <= 500) &
        (F.col("dist_miles") >= 0.1) & (F.col("dist_miles") <= 100)
    )

    duration_expr = (F.unix_timestamp("dropoff_ts") - F.unix_timestamp("pickup_ts")) / 60.0
    weekend_expr = F.when(F.dayofweek("pickup_ts").isin([1, 7]), 1).otherwise(0)

    df = df.withColumn("duration_mins", duration_expr) \
           .withColumn("is_weekend", weekend_expr)

    df = df.filter((F.col("duration_mins") >= 1) & (F.col("duration_mins") <= 240))

    df = df.select(
        F.col("duration_mins"),
        (F.col("dist_miles") / (F.col("duration_mins") / 60.0)).alias("speed_mph"),
        F.col("is_weekend"),
        (F.col("fare_amt") / 500.0).alias("fare_scaled"),
        (F.col("dist_miles") / 100.0).alias("dist_scaled")
    )
    
    df = df.withColumn("speed_scaled", F.col("speed_mph") / 100.0)

    df = df.cache()
    print(f"  Rows Processed: {df.count()}")
    return df

def calculate_stats(df, config):
    print("  Calculating stats...")
    columns = ["duration_mins", "speed_mph", "is_weekend", "fare_scaled", "dist_scaled", "speed_scaled"]
    
    exprs = [F.count(F.lit(1)).alias("total_rows")]
    for c in columns:
        exprs.append(F.mean(c).alias(f"{c}_avg"))
        exprs.append(F.min(c).alias(f"{c}_min"))
        exprs.append(F.max(c).alias(f"{c}_max"))
        
    result = df.agg(*exprs).collect()[0].asDict()
    
    clean_result = {}
    for k, v in result.items():
        clean_result[k] = float(v) if v is not None else 0.0
        if k == "total_rows": clean_result[k] = int(v)

    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config["name"]}.json')
    with open(output_path, 'w') as f:
        json.dump([clean_result], f, indent=2)
    return df

def make_stat(config, step, t0, t1, c0, c1, pipeline_start, mem=0):
    return {
        'config_name': config['name'], 'step': step,
        'start_time': t0 - pipeline_start, 'end_time': t1 - pipeline_start,
        'execution_time_s': t1 - t0, 
        'cpu_time_s': (c1.user - c0.user) + (c1.system - c0.system),
        'peak_memory_mib': mem, 'file_type': 'parquet',
        'num_threads': config.get('num_threads'), 'data_size_pct': config.get('data_size_pct'),
        'system_size_pct': config.get('system_size_pct')
    }

def run_optimized_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Pipeline: {config['name']} (Spark)...")
    start_time = time.perf_counter()
    spark = None
    df = None
    run_stats = []
    
    try:
        t0 = time.perf_counter(); c0 = psutil.Process().cpu_times()
        spark = connect_to_db(config)
        t1 = time.perf_counter(); c1 = psutil.Process().cpu_times()
        run_stats.append(make_stat(config, 'connect_to_db', t0, t1, c0, c1, start_time))

        def _exec(): return execute_deterministic_pipeline(spark, config)

        t0 = time.perf_counter(); c0 = psutil.Process().cpu_times()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem_usage_val, df = memory_usage((_exec, (), {}), max_usage=True, retval=True, interval=0.1)
        t1 = time.perf_counter(); c1 = psutil.Process().cpu_times()
        run_stats.append(make_stat(config, 'execute_deterministic_pipeline', t0, t1, c0, c1, start_time, mem=mem_usage_val))

        t0 = time.perf_counter(); c0 = psutil.Process().cpu_times()
        calculate_stats(df, config)
        t1 = time.perf_counter(); c1 = psutil.Process().cpu_times()
        run_stats.append(make_stat(config, 'calculate_stats', t0, t1, c0, c1, start_time))

    except Exception as e:
        print(f"!!! Error: {e}"); import traceback; traceback.print_exc()
    finally:
        if df: df.unpersist()
        if spark: spark.stop()

    return pd.concat([global_results_df, pd.DataFrame(run_stats)], ignore_index=True)

if __name__ == "__main__":
    SPARK_MODE = os.environ.get("SPARK_MODE", "local")
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        RESULTS_DIR = f"results/local_test/nyc_taxi/spark_{SPARK_MODE}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")
    files = download_taxi_data(get_year_month_list(2009, 1, 2024, 12), LOCAL_DATA_DIR)
    
    if not files: sys.exit(1)

    try:
        total_mem_bytes = psutil.virtual_memory().total
        total_mem_gb = total_mem_bytes / (1024**3)
        total_threads = psutil.cpu_count(logical=True)
        print(f"System detected: {total_threads} threads, {total_mem_gb:.2f} GB RAM")
    except Exception:
        print("Could not detect system info.")
        total_threads = 4
        total_mem_gb = 8

    total_file_count = len(files)
    
    data_size_configs = {
        '100%': files
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
                "memory_limit": sys_config['memory'],
                "num_threads": sys_config['threads'],
                "data_size_pct": data_pct,
                "system_size_pct": sys_pct,
            })
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_optimized_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, f'spark_{SPARK_MODE}_results.csv')
    all_results_df.to_csv(output_csv, index=False)