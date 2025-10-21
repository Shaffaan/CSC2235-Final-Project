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

    execution_log_df = (
        conn.read_csv(str(log_path))
        .select("""
            library,
            replace(step, library||'_', '') as step,
            round(wall_time_sec, 2) as execution_time,
            timestamp,
            row_number() over (partition by step order by timestamp) as iteration_id
        """)
        .filter("step not like '%_load_data' AND step not like '%_create_macros'")
        .filter("iteration_id <= 1")
        .order("iteration_id, timestamp")
        .to_df()
    )
    
    if execution_log_df.empty:
        print("Log file is empty or missing relevant steps. Cannot generate plot.")
        return

    fig = px.bar(
        execution_log_df,
        x="step",
        y="execution_time",
        color="library",
        barmode="group",
        labels={
            "step": "Benchmark Step",
            "execution_time": "Wall Time (seconds)",
            "library": "Library"
        },
        title="Data Preprocessing Pipeline Benchmark (Wall Time)",
    )
    
    plot_path = directory_path / "benchmark_comparison.html"
    fig.write_html(str(plot_path))
    print(f"Plot saved to {plot_path}")