import pandas as pd
import psutil
import time
import os
import sys
import glob
import urllib.request
import math
import warnings
from memory_profiler import memory_usage
import json

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.conf import SparkConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from common.download_utils import get_year_month_list, download_taxi_data

def calculate_and_save_stats(df, config):
    print("  Calculating final statistics...")
    
    columns_to_check = [
        "trip_duration_mins", "speed_mph", "is_weekend",
        "fare_scaled", "dist_scaled", "speed_scaled"
    ]
    
    aggs = [F.count(F.lit(1)).alias("total_rows")]
    for col in columns_to_check:
        aggs.append(F.sum(col).alias(f"{col}_sum"))
        aggs.append(F.avg(col).alias(f"{col}_avg"))
        aggs.append(F.min(col).alias(f"{col}_min"))
        aggs.append(F.max(col).alias(f"{col}_max"))
        aggs.append(F.sum(F.when(F.col(col).isNull(), 1).otherwise(0)).alias(f"{col}_nulls"))

    stats_row = df.agg(*aggs).collect()[0]
    stats_dict = stats_row.asDict()

    config_name = config.get('name', 'unknown_config')
    output_path = os.path.join(os.environ.get('SCRIPT_RESULTS_DIR'), f'stats_{config_name}.json')
    with open(output_path, 'w') as f:
        json.dump([stats_dict], f, indent=2)
    
    print(f"  Final stats saved to {output_path}")
    return df

def connect_to_db(config):
    """
    Initializes SparkSession. Support Local and Distributed modes.
    """
    mem = config.get('memory_limit', '8g')
    if "GB" in mem: mem = mem.replace("GB", "g")
    
    spark_mode = os.environ.get("SPARK_MODE", "local")
    master_url = os.environ.get("SPARK_MASTER_URL")
    
    builder = SparkSession.builder.appName(f"NYC_Taxi_Benchmark_{spark_mode}")
    
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

    builder = builder.config("spark.sql.parquet.mergeSchema", "true")
    
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    return session

def load_data(con, config):
    spark = con
    data_files = config['data_files']
    if not data_files:
        raise ValueError("No data files specified in config['data_files']")
    print(f"  Loading and unifying schema for {len(data_files)} files...")

    files = [f"file://{os.path.abspath(f)}" for f in data_files]

    df = spark.read.parquet(*files)

    existing_cols = set(df.columns)

    schema_map = {
        'tpep_pickup_datetime':  ['tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime'],
        'tpep_dropoff_datetime': ['tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime'],
        'fare_amount':           ['fare_amount', 'Fare_Amt'],
        'trip_distance':         ['trip_distance', 'Trip_Distance'],
        'payment_type':          ['payment_type', 'Payment_Type']
    }

    select_expressions = []
    all_source_cols_to_drop = set()

    for canonical_name, source_options in schema_map.items():
        cols_that_exist = [col for col in source_options if col in existing_cols]
        
        if not cols_that_exist:
            select_expressions.append(F.lit(None).alias(canonical_name))
        else:
            coalesce_exprs = [F.col(c) for c in cols_that_exist]
            select_expressions.append(F.coalesce(*coalesce_exprs).alias(canonical_name))
            all_source_cols_to_drop.update(cols_that_exist)

    cols_to_keep = [c for c in existing_cols if c not in all_source_cols_to_drop]
    
    df_unified = df.select(*cols_to_keep, *select_expressions)

    df_unified = df_unified.cache()
    count = df_unified.count()

    print(f"  Loaded and unified {count} rows.")
    return df_unified


def handle_missing_values(df, config):
    print("  Imputing missing values...")
    
    mean_fare = df.filter(F.col("fare_amount") > 0).select(F.avg("fare_amount")).first()[0] or 0.0
    mean_distance = df.filter(F.col("trip_distance") > 0).select(F.avg("trip_distance")).first()[0] or 0.0

    df_clean = df.withColumn(
        "fare_amount_imputed", 
        F.when((F.col("fare_amount").isNull()) | (F.col("fare_amount") <= 0), F.lit(mean_fare))
         .otherwise(F.col("fare_amount"))
    ).withColumn(
        "trip_distance_imputed",
        F.when((F.col("trip_distance").isNull()) | (F.col("trip_distance") <= 0), F.lit(mean_distance))
         .otherwise(F.col("trip_distance"))
    ).withColumn(
        "payment_type_imputed",
        F.expr("COALESCE(TRY_CAST(payment_type AS INTEGER), 0)")
    )
    
    return df_clean

def feature_engineering(df, config):
    print("  Engineering features...")

    pickup_ts = F.to_timestamp(F.col("tpep_pickup_datetime"))
    dropoff_ts = F.to_timestamp(F.col("tpep_dropoff_datetime"))
    
    df_features = df.withColumn(
        "trip_duration_mins", 
        (dropoff_ts.cast("long") - pickup_ts.cast("long")) / 60.0
    ).withColumn(
        "is_weekend",
        F.dayofweek(pickup_ts).isin([1, 7]).cast("integer")
    )

    df_features = df_features.withColumn(
        "trip_duration_mins",
        F.when((F.col("trip_duration_mins").isNull()) | (F.col("trip_duration_mins") < 0), 0.0)
         .otherwise(F.col("trip_duration_mins"))
    ).withColumn(
        "is_weekend",
        F.coalesce(F.col("is_weekend"), F.lit(0))
    )
    
    df_features = df_features.withColumn(
        "speed_mph",
        F.when(F.col("trip_duration_mins") > 0, 
               F.col("trip_distance_imputed") / (F.col("trip_duration_mins") / 60.0))
         .otherwise(0.0)
    )
    
    # Clean infinities
    df_features = df_features.withColumn(
        "speed_mph",
        F.when(F.isnan(F.col("speed_mph")) | F.col("speed_mph").isin(float('inf'), float('-inf')), 0.0)
         .otherwise(F.col("speed_mph"))
    )
    
    return df_features

def categorical_encoding(df, config):
    payment_types = [r[0] for r in df.select("payment_type_imputed").distinct().collect()]
    print(f"  One-hot encoding for payment types: {payment_types}")
    
    select_clauses = []
    for pt in payment_types:
        pt_str = str(pt).replace('.', '_').replace('-', '_')
        clause = F.when(F.col("payment_type_imputed") == pt, 1).otherwise(0).alias(f"payment_type_{pt_str}")
        select_clauses.append(clause)
        
    if not select_clauses:
        return df
    
    df_encoded = df.select("*", *select_clauses)
    return df_encoded

def numerical_scaling(df, config):
    print("  [INTERMEDIATE] Aggregating Min/Max for Scaling...")
    params = df.agg(
        F.min("fare_amount_imputed").alias("min_fare"),
        F.max("fare_amount_imputed").alias("max_fare"),
        F.min("trip_distance_imputed").alias("min_dist"),
        F.max("trip_distance_imputed").alias("max_dist"),
        F.min("speed_mph").alias("min_speed"),
        F.max("speed_mph").alias("max_speed")
    ).first()

    min_fare = params["min_fare"] or 0
    max_fare = params["max_fare"] or 0
    min_dist = params["min_dist"] or 0
    max_dist = params["max_dist"] or 0
    min_speed = params["min_speed"] or 0
    max_speed = params["max_speed"] or 0

    print(f"  Scaling params: Fare({min_fare}, {max_fare}), Dist({min_dist}, {max_dist}), Speed({min_speed}, {max_speed})")

    def min_max_scaler(col_name, min_val, max_val):
        denominator = max_val - min_val
        if denominator == 0: return F.lit(0.0)
        return (F.col(col_name) - min_val) / denominator
        
    df_scaled = df.withColumn("fare_scaled", min_max_scaler("fare_amount_imputed", min_fare, max_fare)) \
                  .withColumn("dist_scaled", min_max_scaler("trip_distance_imputed", min_dist, max_dist)) \
                  .withColumn("speed_scaled", min_max_scaler("speed_mph", min_speed, max_speed))
                  
    return df_scaled

def run_full_pipeline(config, global_results_df):
    print(f"\n[BEGIN] Running pipeline: {config['name']} (Spark Optimized)...")
    run_results_list = []
    pipeline_start_time = time.perf_counter()
    
    spark_session = None
    df = None
    step_name = "N/A"
    
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
            
            if i == 0: 
                current_args = (config,)
            elif i == 1: 
                if spark_session is None: break
                current_args = (spark_session, config)
            else: 
                if df is None: break
                current_args = (df, config)
                
            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_times_before = process.cpu_times()
            step_start_time = time.perf_counter()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mem_usage = memory_usage((func, current_args, kwargs), max_usage=True, retval=True, interval=0.1)
            
            peak_mem_mib = mem_usage[0]
            result = mem_usage[1]
            step_end_time = time.perf_counter()
            cpu_times_after = process.cpu_times()
            
            if i == 0:
                spark_session = result
            else:
                df = result
                if i > 1 and step_name != "calculate_and_save_stats":
                     df = df.cache()
                     df.count() 

            elapsed_time = step_end_time - step_start_time
            cpu_time_used = (cpu_times_after.user - cpu_times_before.user) + (cpu_times_after.system - cpu_times_before.system)
            abs_start_time = step_start_time - pipeline_start_time
            abs_end_time = step_end_time - pipeline_start_time
            
            print(f"  [END] Step: {step_name} ({elapsed_time:.4f}s)")
            
            run_results_list.append({
                'config_name': config.get('name', 'N/A'),
                'step': step_name,
                'start_time': abs_start_time, 'end_time': abs_end_time,
                'execution_time_s': elapsed_time, 'cpu_time_s': cpu_time_used,
                'peak_memory_mib': peak_mem_mib, 'file_type': config.get('file_type', 'N/A'),
                'memory_limit': config.get('memory_limit', 'None'),
                'num_threads': config.get('num_threads', 'N/A'),
                'data_size_pct': config.get('data_size_pct', 'N/A'),
                'system_size_pct': config.get('system_size_pct', 'N/A')
            })
    except Exception as e:
        print(f"!!! ERROR in pipeline config '{config['name']}' at step '{step_name}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if spark_session:
            spark_session.stop()
        del df
        
    print(f"[END] Pipeline: {config['name']} finished.")
    run_results_df = pd.DataFrame(run_results_list)
    return pd.concat([global_results_df, run_results_df], ignore_index=True)


if __name__ == "__main__":
    SPARK_MODE = os.environ.get("SPARK_MODE", "local")
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        RESULTS_DIR = f"results/local_test/nyc_taxi/spark_{SPARK_MODE}"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    START_YEAR, START_MONTH = 2009, 1
    END_YEAR, END_MONTH = 2024, 12
    
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")
    year_month_list = get_year_month_list(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    all_files_sorted = download_taxi_data(year_month_list, LOCAL_DATA_DIR)
    
    if not all_files_sorted:
        sys.exit(1)

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
            config_name = f"Data_{data_pct}_Sys_{sys_pct}"
            CONFIGURATIONS.append({
                "name": config_name, "file_type": "parquet", "data_files": file_list,
                "memory_limit": sys_config['memory'], "num_threads": sys_config['threads'],
                "data_size_pct": data_pct, "system_size_pct": sys_pct,
            })
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, f'spark_{SPARK_MODE}_results.csv')
    all_results_df.to_csv(output_csv, index=False)