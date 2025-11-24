import os
import glob
import pandas as pd
import plotly.express as px
import sys

def extract_null_percentage(config_name: str) -> float:
    """
    Extract Null % from strings like:
    'Data_1%_Sys_100%_Null_10%'
    """
    try:
        parts = config_name.split("_")
        if "Null" in parts:
            idx = parts.index("Null")
            pct_str = parts[idx + 1]
            return float(pct_str.replace("%", ""))
    except:
        pass

    return None


def load_all_results(results_dir):
    """
    Walks results_dir and loads all *_results.csv files for each framework.
    Returns a single combined DataFrame.
    """
    dfs = []

    for pipeline_dir in glob.glob(os.path.join(results_dir, "*")):
        if not os.path.isdir(pipeline_dir):
            continue

        for framework_dir in glob.glob(os.path.join(pipeline_dir, "*")):
            if not os.path.isdir(framework_dir):
                continue

            framework = os.path.basename(framework_dir)
            print(framework)
            csv_path = os.path.join(framework_dir, f"{framework}_results.csv")

            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            df["framework"] = framework
            dfs.append(df)

    if not dfs:
        print("No results CSVs found.")
        sys.exit(1)

    return pd.concat(dfs, ignore_index=True)


def generate_plot(df,results_dir, output_file="handle_nulls_time.html"):
    """
    Generates a line graph:
      X-axis = null %
      Y-axis = time
      Lines = pandas, polars, duckdb
    """

    df = df.copy()
    df = df[df["step"] == "handle_missing_values"]

    # Extract null percentage
    df["null_pct"] = df["config_name"].apply(extract_null_percentage)

    # Filter frameworks to standard names (if present)
    df = df[df["framework"].isin(["pandas", "polars", "duckdb"])]
    #print(df)

    # Ensure numeric
    df["execution_time_s"] = pd.to_numeric(df["execution_time_s"])
    df["null_pct"] = df["null_pct"]

    fig = px.line(
        df,
        x="null_pct",
        y="execution_time_s",
        color="framework",
        markers=True,
        title="Handle Nulls Performance vs Null Percentage",
        labels={
            "null_pct": "Null Percentage (%)",
            "execution_time_s": "Execution Time (s)",
            "framework": "Framework"
        }
    )
    output_path = os.path.join(results_dir, output_file)
    fig.write_html(output_path)
    print(f"Plot written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    full_df = load_all_results(results_dir)
    generate_plot(full_df, results_dir)
