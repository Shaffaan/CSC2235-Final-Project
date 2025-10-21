from pathlib import Path
import duckdb
import plotly.express as px
from utils import LOG_FILE

def generate_plots():
    """
    Generates an HTML bar chart from the execution log.
    """
    conn = duckdb.connect()
    directory_path = Path(__file__).resolve().parent
    log_path = directory_path / LOG_FILE

    if not log_path.exists():
        print(f"Log file not found at {log_path}. Cannot generate plot.")
        return

    try:
        # --- Ensure correct reading and filtering ---
        execution_log_df = (
            conn.read_csv(str(log_path))
            .select("""
                library,
                step, -- Keep original step name for filtering
                replace(step, library||'_', '') as step_name, -- Use alias for plotting axis
                round(wall_time_sec, 2) as execution_time,
                timestamp,
                row_number() over (partition by step order by timestamp) as iteration_id
            """)
            .filter("""
                step LIKE '%_load_data' OR
                step LIKE '%_encode_data' OR
                step LIKE '%_split_data' OR
                step LIKE '%_train_model' OR
                step LIKE '%_predict_model'
            """)
            .filter("iteration_id <= 1")
            .order("iteration_id, timestamp")
            .to_df()
        )
    except Exception as e:
        print(f"Error reading or processing log file: {e}")
        return


    if execution_log_df.empty:
        print("Log file is empty *after filtering* or missing relevant steps. Cannot generate plot.")
        return

    fig = px.bar(
        execution_log_df,
        x="step_name",
        y="execution_time",
        color="library",
        barmode="group",
        labels={
            "step_name": "Benchmark Step",
            "execution_time": "Wall Time (seconds)",
            "library": "Library"
        },
        title="ML Pipeline Benchmark (Palmer Penguins)",
        category_orders={
            "step_name": [
                "load_data",
                "encode_data",
                "split_data",
                "train_model",
                "predict_model"
            ]
        }
    )

    plot_path = directory_path / "benchmark_comparison.html"
    fig.write_html(str(plot_path))
    print(f"Plot saved to {plot_path}")