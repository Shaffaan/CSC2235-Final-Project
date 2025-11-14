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
    """
    Calculates aggregate stats and saves them to a JSON file.
    """
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
    Initializes and returns a SparkSession.
    This acts as the 'connection' for the benchmark harness.
    """
    num_threads = config.get('num_threads', '*')
    memory_limit = config.get('memory_limit', '8g')
    
    if isinstance(memory_limit, (int, float)):
        memory_limit = f"{int(memory_limit)}g"
    elif "GB" in memory_limit:
        memory_limit = memory_limit.replace("GB", "g")
        
    conf = SparkConf() \
        .setMaster(f"local[{num_threads}]") \
        .setAppName("NYC_Taxi_Benchmark") \
        .set("spark.driver.memory", memory_limit) \
        .set("spark.sql.parquet.mergeSchema", "true")
        
    session = SparkSession.builder.config(conf=conf).getOrCreate()
    
    print(f"  Connected to Spark. Master: local[{num_threads}], Driver Memory: {memory_limit}")
    return session

def load_data(con, config):
    """
    Loads data using SparkSession 'con'.
    'con' is the SparkSession.
    Returns an eager (cached) Spark DataFrame.
    """
    spark = con
    data_files = config['data_files']
    if not data_files:
        raise ValueError("No data files specified in config['data_files']")
    print(f"  Loading and unifying schema for {len(data_files)} files...")

    df = spark.read.parquet(*data_files)

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

    print(f"  Loaded and unified {count} rows from {len(data_files)} files.")
    return df_unified


def handle_missing_values(df, config):
    """
    Accepts an eager Spark DataFrame 'df'
    and returns a new eager (cached) DataFrame.
    """
    print("  Imputing missing values...")
    
    mean_fare = df.filter(F.col("fare_amount") > 0).select(F.avg("fare_amount")).first()[0]
    mean_fare = mean_fare or 0.0
    
    mean_distance = df.filter(F.col("trip_distance") > 0).select(F.avg("trip_distance")).first()[0]
    mean_distance = mean_distance or 0.0

    fare_cond = (F.col("fare_amount").isNull()) | (F.col("fare_amount") <= 0)
    fare_expr = F.when(fare_cond, F.lit(mean_fare)).otherwise(F.col("fare_amount"))
    
    dist_cond = (F.col("trip_distance").isNull()) | (F.col("trip_distance") <= 0)
    dist_expr = F.when(dist_cond, F.lit(mean_distance)).otherwise(F.col("trip_distance"))

    payment_expr = F.expr("COALESCE(TRY_CAST(payment_type AS INTEGER), 0)")

    df_clean = df.withColumn("fare_amount_imputed", fare_expr) \
                 .withColumn("trip_distance_imputed", dist_expr) \
                 .withColumn("payment_type_imputed", payment_expr)
    
    df_clean = df_clean.cache()
    df_clean.count()
    
    print("  Imputed missing values.")
    return df_clean

def feature_engineering(df, config):
    """
    Performs feature engineering using Spark functions.
    Accepts and returns an eager (cached) DataFrame.
    """
    print("  Engineering features...")

    pickup_ts = F.to_timestamp(F.col("tpep_pickup_datetime"))
    dropoff_ts = F.to_timestamp(F.col("tpep_dropoff_datetime"))
    duration_secs = dropoff_ts.cast("long") - pickup_ts.cast("long")
    duration_mins_expr = duration_secs / 60.0
    
    is_weekend_expr = F.dayofweek(pickup_ts).isin([1, 7]).cast("integer")

    df_features = df.withColumn("trip_duration_mins", duration_mins_expr) \
                    .withColumn("is_weekend", is_weekend_expr)

    df_features = df_features.withColumn(
        "trip_duration_mins",
        F.when(
            F.col("trip_duration_mins").isNull() | (F.col("trip_duration_mins") < 0),
            F.lit(0.0)
        ).otherwise(F.col("trip_duration_mins"))
    ).withColumn(
        "is_weekend",
        F.when(F.col("is_weekend").isNull(), F.lit(0)).otherwise(F.col("is_weekend"))
    )
    
    speed_expr = F.when(
        F.col("trip_duration_mins") > 0,
        F.col("trip_distance_imputed") / (F.col("trip_duration_mins") / 60.0)
    ).otherwise(F.lit(0.0))

    df_features = df_features.withColumn("speed_mph", speed_expr)
    
    df_features = df_features.withColumn(
        "speed_mph",
        F.when(
            F.col("speed_mph").isNull() | 
            F.isnan(F.col("speed_mph")) | 
            (F.col("speed_mph") == F.lit(float('inf'))) | 
            (F.col("speed_mph") == F.lit(float('-inf'))),
            F.lit(0.0)
        ).otherwise(F.col("speed_mph"))
    )
    
    df_features = df_features.cache()
    df_features.count()

    print("  Engineered features: duration, speed, is_weekend.")
    return df_features

def categorical_encoding(df, config):
    """
    Performs one-hot encoding on the 'payment_type_imputed' column.
    Accepts and returns an eager (cached) DataFrame.
    """
    payment_types = [r[0] for r in df.select("payment_type_imputed").distinct().collect()]
    print(f"  One-hot encoding for payment types: {payment_types}")
    
    select_clauses = []
    for pt in payment_types:
        pt_str = str(pt).replace('.', '_').replace('-', '_')
        
        clause = F.when(F.col("payment_type_imputed") == pt, 1).otherwise(0).alias(f"payment_type_{pt_str}")
        select_clauses.append(clause)
        
    if not select_clauses:
        print("  No payment types found to encode. Skipping.")
        return df
    
    df_encoded = df.select("*", *select_clauses)
    
    df_encoded = df_encoded.cache()
    df_encoded.count()

    print("  Performed one-hot encoding.")
    return df_encoded

def numerical_scaling(df, config):
    """
    Applies Min-Max scaling.
    Accepts and returns an eager (cached) DataFrame.
    """
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
        if denominator == 0:
            return F.lit(0.0)
        return (F.col(col_name) - min_val) / denominator
        
    df_scaled = df.withColumn(
        "fare_scaled", 
        min_max_scaler("fare_amount_imputed", min_fare, max_fare)
    ).withColumn(
        "dist_scaled", 
        min_max_scaler("trip_distance_imputed", min_dist, max_dist)
    ).withColumn(
        "speed_scaled", 
        min_max_scaler("speed_mph", min_speed, max_speed)
    )
    
    df_scaled = df_scaled.cache()
    df_scaled.count()
    
    print("  Applied Min-Max scaling.")
    return df_scaled

def run_full_pipeline(config, global_results_df):
    """
    Modified harness to run Spark pipeline.
    It passes the SparkSession to the first step,
    then passes the resulting DataFrame to subsequent steps.
    It also properly stops the SparkSession in the 'finally' block.
    """
    print(f"\n[BEGIN] Running pipeline: {config['name']}...")
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
                if spark_session is None:
                    print("  SparkSession is missing. Stopping pipeline.")
                    break
                current_args = (spark_session, config)
            else: 
                if df is None:
                    print("  DataFrame is missing. Stopping pipeline.")
                    break
                current_args = (df, config)
                
            print(f"  [START] Step: {step_name}")
            process = psutil.Process(os.getpid())
            cpu_times_before = process.cpu_times()
            step_start_time = time.perf_counter()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                mem_usage = memory_usage((func, current_args, kwargs), max_usage=True, retval=True, interval=0.1)
            
            peak_mem_mib = mem_usage[0]
            result = mem_usage[1]
            step_end_time = time.perf_counter()
            cpu_times_after = process.cpu_times()
            
            if i == 0:
                spark_session = result
            else:
                if df is not None:
                    df.unpersist()
                df = result

            elapsed_time = step_end_time - step_start_time
            cpu_time_used = (cpu_times_after.user - cpu_times_before.user) + (cpu_times_after.system - cpu_times_before.system)
            abs_start_time = step_start_time - pipeline_start_time
            abs_end_time = step_end_time - pipeline_start_time
            
            print(f"  [END] Step: {step_name}")
            print(f"    Exec Time: {elapsed_time:.4f} s")
            print(f"    CPU Time: {cpu_time_used:.4f} s")
            print(f"    Peak Mem: {peak_mem_mib:.2f} MiB")
            
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
    finally:
        if spark_session:
            try:
                spark_session.stop()
                print("  SparkSession stopped.")
            except Exception as ce:
                print(f"  Error while stopping SparkSession: {ce}", file=sys.stderr)
        del df
        
    print(f"[END] Pipeline: {config['name']} finished.")
    run_results_df = pd.DataFrame(run_results_list)
    return pd.concat([global_results_df, run_results_df], ignore_index=True)


if __name__ == "__main__":
    RESULTS_DIR = os.environ.get('SCRIPT_RESULTS_DIR') 
    if not RESULTS_DIR:
        print("Error: SCRIPT_RESULTS_DIR environment variable not set. Run this via run.sh")
        RESULTS_DIR = "results/local_test/nyc_taxi/spark"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    START_YEAR, START_MONTH = 2009, 1
    END_YEAR, END_MONTH = 2025, 9
    
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")
    
    year_month_list = get_year_month_list(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    
    all_files_sorted = download_taxi_data(year_month_list, LOCAL_DATA_DIR)
    
    if not all_files_sorted:
        print("No data files found or downloaded. Exiting benchmark.")
        sys.exit(1)

    try:
        total_mem_bytes = psutil.virtual_memory().total
        total_mem_gb = total_mem_bytes / (1024**3)
        total_cores = psutil.cpu_count(logical=True)
        print(f"System detected: {total_cores} logical cores, {total_mem_gb:.2f} GB RAM")
    except Exception:
        print("Could not detect system info, using defaults (4 cores, 8GB RAM).")
        total_mem_gb = 8.0
        total_cores = 4
    
    safe_max_mem_gb = total_mem_gb * 0.75
    total_file_count = len(all_files_sorted)
    
    data_size_configs = {
        '1%': all_files_sorted[:max(1, int(total_file_count * 0.01))],
        '2%': all_files_sorted[:max(1, int(total_file_count * 0.02))],
        # '30%': all_files_sorted[:max(1, int(total_file_count * 0.30))],
        # '100%': all_files_sorted,
    }
    system_size_configs = {
        # '10%': {'threads': max(1, int(total_cores * 0.10)), 'memory': f"{max(1, int(safe_max_mem_gb * 0.10))}GB"},
        # '30%': {'threads': max(1, int(total_cores * 0.30)), 'memory': f"{max(1, int(safe_max_mem_gb * 0.30))}GB"},
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
    
    print(f"\n--- Generated {len(CONFIGURATIONS)} test configurations for Spark ---")
    
    all_results_df = pd.DataFrame()
    for config in CONFIGURATIONS:
        all_results_df = run_full_pipeline(config, all_results_df)

    output_csv = os.path.join(RESULTS_DIR, 'spark_results.csv')
    all_results_df.to_csv(output_csv, index=False)
    
    print(f"\n--- Spark benchmarks complete ---")
    if not all_results_df.empty:
        print(all_results_df[['config_name', 'step', 'execution_time_s', 'peak_memory_mib']])
    else:
        print("No results were generated.")
    print(f"Full results saved to {output_csv}")