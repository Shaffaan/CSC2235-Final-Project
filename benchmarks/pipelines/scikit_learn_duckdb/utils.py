import time
import csv
import functools
import os
import tracemalloc
import psutil
import requests
import duckdb
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession

# --- File Caching Vars ---
LOCAL_CSV = "penguins.csv"
DATA_URL = "http://blobs.duckdb.org/data/penguins.csv"
LOG_FILE = "execution_log.csv"

# --- Uniform split files ---
TRAIN_IDS_PARQUET = "train_ids.parquet"
TEST_IDS_PARQUET = "test_ids.parquet"

# Get a handle on the current process
PROC = psutil.Process(os.getpid())

def timeit(method):
    """
    A decorator to measure wall time, CPU time, and memory usage 
    of a function, logging results to a CSV.
    """
    @functools.wraps(method)
    def timed(*args, **kwargs):
        
        # --- Start resource tracking ---
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            cpu_start = PROC.cpu_times()
        except psutil.NoSuchProcess:
            cpu_start = None

        # --- Execute Function ---
        result = method(*args, **kwargs)
        
        # --- Stop resource tracking ---
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        try:
            cpu_end = PROC.cpu_times()
            if cpu_start:
                cpu_user = cpu_end.user - cpu_start.user
                cpu_system = cpu_end.system - cpu_start.system
                cpu_total = cpu_user + cpu_system
            else:
                cpu_total = 0
        except psutil.NoSuchProcess:
            cpu_total, cpu_user, cpu_system = 0, 0, 0
            
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        try:
            mem_rss = PROC.memory_info().rss
        except psutil.NoSuchProcess:
            mem_rss = 0
        
        # --- CSV Logging ---
        method_source = method.__name__.split("_")[0]
        with open(LOG_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                method_source,
                method.__name__,
                f"{elapsed_time:.4f}",
                f"{cpu_total:.4f}",
                f"{mem_peak / 1024**2:.2f}",
                f"{mem_rss / 1024**2:.2f}"
            ])
            
        print(f"--- Function: {method.__name__} ---")
        print(f"  Wall Time:  {elapsed_time:.4f} seconds")
        print(f"  CPU Time:   {cpu_total:.4f} seconds (User: {cpu_user:.4f}s, System: {cpu_system:.4f}s)")
        print(f"  Peak Heap:  {mem_peak / 1024**2:.2f} MB")
        print(f"  Final RSS:  {mem_rss / 1024**2:.2f} MB")
        print("-" * (len(method.__name__) + 20))
        
        return result
    return timed

def _download_file():
    """Helper function to download the penguins.csv file."""
    print(f"Local file not found. Downloading from {DATA_URL}...")
    try:
        with requests.get(DATA_URL, stream=True) as r:
            r.raise_for_status()
            with open(LOCAL_CSV, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Download complete. Saved as {LOCAL_CSV}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(LOCAL_CSV):
            os.remove(LOCAL_CSV)
        raise e

def setup_data_source():
    """
    Ensures the penguins.csv file exists, downloading if needed.
    """
    print("--- Setting up data source ---")
    if os.path.exists(LOCAL_CSV):
        print(f"Data file found: {LOCAL_CSV}")
        return

    _download_file()
    print("-" * 30)

def create_uniform_split_indexes():
    """
    Creates and saves a uniform 70/30 train/test split of observation IDs
    to ensure all libraries use the exact same data.
    """
    print("--- Creating uniform train/test split IDs ---")
    if os.path.exists(TRAIN_IDS_PARQUET) and os.path.exists(TEST_IDS_PARQUET):
        print("Train/test ID files already exist.")
        print("-" * 30)
        return

    print("Generating new train/test ID files (70/30 split)...")
    try:
        # Use DuckDB to quickly load, filter, and add IDs
        conn = duckdb.connect()
        (
            conn.read_csv(LOCAL_CSV)
            .filter("columns(*)::text != 'NA'")
            .filter("columns(*) is not null")
            .select("row_number() over () as observation_id")
        ).to_table("all_ids")
        
        # Create a reproducible 70% sample
        conn.sql(f"CREATE TABLE train_ids AS SELECT * FROM all_ids USING SAMPLE 70 PERCENT (reservoir, 256)")
        
        # Get the remaining 30%
        conn.sql(f"""
            CREATE TABLE test_ids AS
            SELECT observation_id FROM all_ids
            WHERE observation_id NOT IN (SELECT observation_id FROM train_ids)
        """)
        
        # Save them to Parquet
        train_ids = conn.table("train_ids").to_df()
        test_ids = conn.table("test_ids").to_df()
        
        train_ids.to_parquet(TRAIN_IDS_PARQUET)
        test_ids.to_parquet(TEST_IDS_PARQUET)
        
        conn.close()
        
        print(f"Saved {len(train_ids)} train IDs to {TRAIN_IDS_PARQUET}")
        print(f"Saved {len(test_ids)} test IDs to {TEST_IDS_PARQUET}")
        
    except Exception as e:
        print(f"Error creating uniform split IDs: {e}")
        raise e
    print("-" * 30)


def clear_log_file():
    """Clears the log file and writes the header."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    with open(LOG_FILE, "x") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "library", "step", "wall_time_sec", 
            "cpu_time_sec", "peak_heap_mb", "final_rss_mb"
        ])