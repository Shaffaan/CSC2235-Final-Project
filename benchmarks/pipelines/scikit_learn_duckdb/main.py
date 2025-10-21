import os
import sys
from pyspark.sql import SparkSession
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add current dir to path to ensure modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import utils
    import run_benchmarks
    import reconcile
    import plot
except ImportError as e:
    print(f"Error: Failed to import a required module. {e}")
    print("Please ensure all files (utils.py, run_benchmarks.py, reconcile.py, plot.py) are in the same directory.")
    sys.exit(1)


def main():
    """
    Main orchestrator script to run all ML benchmarks, reconcile, and plot.
    """
    # --- 1. Setup ---
    print("--- 1. Initializing Benchmark ---")
    utils.clear_log_file()
    utils.setup_data_source()
    utils.create_uniform_split_indexes()
    
    results = {}
    
    # --- 2. Run Benchmarks ---
    print("\n--- 2. Running Benchmarks ---")
    
    try:
        print("\n--- Running DuckDB Benchmark ---")
        results['duckdb'] = run_benchmarks.run_duckdb_benchmark()
    except Exception as e:
        print(f"[FAIL] DuckDB benchmark failed: {e}")
        results['duckdb'] = {} # Add empty dict to avoid key errors

    try:
        print("\n--- Running Pandas Benchmark ---")
        results['pandas'] = run_benchmarks.run_pandas_benchmark()
    except Exception as e:
        print(f"[FAIL] Pandas benchmark failed: {e}")
        results['pandas'] = {}

    try:
        print("\n--- Running Polars Benchmark ---")
        results['polars'] = run_benchmarks.run_polars_benchmark()
    except Exception as e:
        print(f"[FAIL] Polars benchmark failed: {e}")
        results['polars'] = {}

    try:
        print("\n--- Running Scikit-learn Benchmark ---")
        results['scikit'] = run_benchmarks.run_scikit_benchmark()
    except Exception as e:
        print(f"[FAIL] Scikit-learn benchmark failed: {e}")
        results['scikit'] = {}

    try:
        print("\n--- Running Spark Benchmark ---")
        print("Initializing Spark session...")
        spark = (
            SparkSession.builder
            .appName("MLBenchmark")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.log.level", "ERROR")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        
        results['spark'] = run_benchmarks.run_spark_benchmark(spark)
        
        print("Stopping Spark session...")
        spark.stop()
        
    except Exception as e:
        print(f"[FAIL] Spark benchmark failed: {e}")
        results['spark'] = {}
        try:
            spark.stop()
        except:
            pass

    print("\n--- All benchmarks complete. ---")

    # --- 3. Reconcile Results ---
    print("\n--- 3. Reconciling Results ---")
    try:
        reconcile.reconcile_all_results(results)
    except Exception as e:
        print(f"[FAIL] Reconciliation failed: {e}")

    # --- 4. Generate Plots ---
    print("\n--- 4. Generating Plots ---")
    try:
        plot.generate_plots()
    except Exception as e:
        print(f"[FAIL] Plotting failed: {e}")

    print("\n--- Benchmark Suite Finished ---")

if __name__ == "__main__":
    main()