# Data Pipeline Benchmarking Harness

This project is a fair benchmarking harness for comparing data processing pipelines. It is designed to run the same ETL logic (load, clean, feature engineer, scale) across multiple frameworks on ML pipelines.

The currently supported frameworks are:

  * DuckDB
  * Polars
  * Pandas
  * PySpark

## Running the Benchmarks

The `run_benchmarks.sh` script is the main entry point. It handles clearing the OS cache (for fair disk I/O) and executing the Python pipeline scripts.

**Run all pipelines and all frameworks:**

```bash
./run_benchmarks.sh
```

**Run all frameworks for a single pipeline:**

```bash
./run_benchmarks.sh nyc_taxi
```

**Run a specific framework for a specific pipeline:**

```bash
./run_benchmarks.sh nyc_taxi polars
```

## 📊 Results

All results are saved to a new timestamped directory inside `results/`.

After the benchmarks run, the `common/summarize_results.py` script is automatically executed to generate a final report, which includes:

1.  **Performance Plots:** Interactive HTML plots (`.html`) are generated in the results directory, comparing:

      * Step-by-step execution timelines for each framework.
      * Peak memory usage for each framework.
      * Individual framework configuration timelines.

2.  **Reconciliation Report:** A check is printed to your console that validates the final aggregate statistics (`stats_...json`) from each framework to ensure they all produced the same numerical results.