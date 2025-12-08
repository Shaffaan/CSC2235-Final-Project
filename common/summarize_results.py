import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys
import os
import glob
import numpy as np
import json

PREFERRED_FRAMEWORK_ORDER = ['duckdb', 'polars', 'pandas', 'spark_local', 'spark_distributed']

def visualize_config_comparison(results_df, output_file):
    if results_df.empty: return
    pio.templates.default = "plotly_dark"
    
    cols_to_numeric = ['start_time', 'execution_time_s', 'end_time']
    for c in cols_to_numeric:
        results_df[c] = pd.to_numeric(results_df[c], errors='coerce')

    fig = px.bar(
        results_df, base="start_time", x="execution_time_s", y="config_name",
        color="step", orientation='h', title="Framework Config Comparison",
        hover_data=['execution_time_s', 'peak_memory_mib', 'cpu_time_s', 'start_time', 'end_time']
    )
    fig.update_yaxes(autorange="reversed")
    fig.write_html(output_file)

def visualize_framework_comparison(all_results_df, output_dir):
    combinations = all_results_df[['data_size_pct', 'system_size_pct']].drop_duplicates()
    for index, row in combinations.iterrows():
        data_pct, sys_pct = row['data_size_pct'], row['system_size_pct']
        current_combo_df = all_results_df.loc[
            (all_results_df['data_size_pct'] == data_pct) & 
            (all_results_df['system_size_pct'] == sys_pct)
        ].copy()
        
        if current_combo_df.empty: continue
        
        # Timeline Plot
        try:
            for c in ['start_time', 'execution_time_s', 'end_time']:
                current_combo_df[c] = pd.to_numeric(current_combo_df[c])
            
            fig = px.bar(
                current_combo_df, base="start_time", x="execution_time_s", y="framework",
                color="step", orientation='h', 
                title=f'Pipeline Timeline ({data_pct} Data, {sys_pct} System)',
                category_orders={'framework': PREFERRED_FRAMEWORK_ORDER}
            )
            filename = f"timeline_{str(data_pct).replace('%','pct')}_{str(sys_pct).replace('%','pct')}.html"
            fig.write_html(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"    Plot error: {e}")

def reconcile_stats(full_stats_df, pipeline_name):
    print(f"\n--- 📊 Reconciling Stats: {pipeline_name} ---")
    all_match = True
    for config in full_stats_df['config_name'].unique():
        stats_df = full_stats_df[full_stats_df['config_name'] == config].reset_index(drop=True)
        if len(stats_df) < 2: continue
        
        baseline = stats_df.iloc[0]
        config_match = True
        
        print(f"  Config {config}: Comparing against {baseline['framework']}")
        for idx, row in stats_df.iloc[1:].iterrows():
            for col in stats_df.columns:
                if col in ['framework', 'config_name']: continue
                
                b_val, c_val = baseline[col], row[col]
                
                is_match = False
                if pd.api.types.is_number(b_val) and pd.api.types.is_number(c_val):
                    if np.isclose(b_val, c_val, rtol=1e-5, equal_nan=True): is_match = True
                else:
                    if b_val == c_val or (pd.isna(b_val) and pd.isna(c_val)): is_match = True
                
                if not is_match:
                    print(f"    ❌ Mismatch {row['framework']} on {col}: {b_val} vs {c_val}")
                    config_match = False
                    all_match = False
        
        if config_match: print("    ✅ All Match")

def summarize_cv_pipeline(results_root):
    print(f"[CV Summary] Scanning {results_root}")
    summary_dir = os.path.join(results_root, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    bucket_frames, timing_frames = [], []
    
    for root, dirs, files in os.walk(results_root):
        if "bucket_summary.csv" in files:
            pipeline_name = os.path.basename(os.path.dirname(root))
            run_label = os.path.basename(root)
            
            df = pd.read_csv(os.path.join(root, "bucket_summary.csv"))
            df['pipeline'] = "cv"
            df['framework'] = pipeline_name if "cv" not in pipeline_name else run_label
            bucket_frames.append(df)
            
        if "step_timings.csv" in files:
            t_df = pd.read_csv(os.path.join(root, "step_timings.csv"))
            t_df['framework'] = os.path.basename(root)
            timing_frames.append(t_df)

    if bucket_frames:
        combined_buckets = pd.concat(bucket_frames, ignore_index=True)
        combined_buckets.to_csv(os.path.join(summary_dir, "cv_bucket_summary_combined.csv"), index=False)
        print(f"  Saved CV bucket summary to {summary_dir}")

    if timing_frames:
        combined_timing = pd.concat(timing_frames, ignore_index=True)
        combined_timing.to_csv(os.path.join(summary_dir, "cv_timings_combined.csv"), index=False)
        
        combined_timing['total_time'] = combined_timing.groupby(['framework'])['duration_s'].transform('sum')
        fig = px.bar(combined_timing, x='total_time', y='framework', title="CV Pipeline Duration")
        fig.write_html(os.path.join(summary_dir, "cv_runtime_plot.html"))
        print(f"  Saved CV runtime plot to {summary_dir}")

def summarize_wikipedia_pipeline(results_root):
    print(f"[Wikipedia Summary] Scanning {results_root}")
    join_metrics = []
    
    for root, dirs, files in os.walk(results_root):
        for f in files:
            if f.startswith("join_metrics") and f.endswith(".csv"):
                framework = os.path.basename(root)
                df = pd.read_csv(os.path.join(root, f))
                df['framework'] = framework
                join_metrics.append(df)

    if not join_metrics:
        print("  No Wikipedia join metrics found.")
        return

    combined = pd.concat(join_metrics, ignore_index=True)
    plots_dir = os.path.join(results_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    summary = combined.groupby(["config_name", "join_type", "framework"])['execution_time_s'].mean().reset_index()
    summary.to_csv(os.path.join(plots_dir, "wikipedia_join_summary.csv"), index=False)
    
    for join_type in combined['join_type'].unique():
        subset = summary[summary['join_type'] == join_type]
        fig = px.bar(
            subset, x='config_name', y='execution_time_s', color='framework', barmode='group',
            title=f"Wikipedia Join Performance: {join_type}"
        )
        fig.write_html(os.path.join(plots_dir, f"join_{join_type}.html"))
    
    print(f"  Saved Wikipedia summaries to {plots_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <timestamped_results_dir>")
        sys.exit(1)
        
    TOP_LEVEL_RESULTS_DIR = sys.argv[1]
    
    if not os.path.isdir(TOP_LEVEL_RESULTS_DIR):
        print(f"Error: Directory not found: {TOP_LEVEL_RESULTS_DIR}")
        sys.exit(1)

    print(f"Summarizing results in: {TOP_LEVEL_RESULTS_DIR}")
    
    subdirs = os.listdir(TOP_LEVEL_RESULTS_DIR)
    
    if "cv" in subdirs:
        summarize_cv_pipeline(os.path.join(TOP_LEVEL_RESULTS_DIR, "cv"))
        
    if "wikipedia_data" in subdirs:
        summarize_wikipedia_pipeline(os.path.join(TOP_LEVEL_RESULTS_DIR, "wikipedia_data"))
    
    for pipeline_dir in glob.glob(os.path.join(TOP_LEVEL_RESULTS_DIR, '*')):
        if not os.path.isdir(pipeline_dir): continue
        pipeline_name = os.path.basename(pipeline_dir)
        
        if pipeline_name in ["cv", "wikipedia_data", "plots", "summaries"]: continue
        
        print(f"\nProcessing Tabular Pipeline: {pipeline_name}")
        
        all_framework_dfs = []
        all_framework_stats = []
        
        for framework_dir in glob.glob(os.path.join(pipeline_dir, '*')):
            if not os.path.isdir(framework_dir): continue
            fw_name = os.path.basename(framework_dir)
            
            csv_path = os.path.join(framework_dir, f"{fw_name}_results.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df['framework'] = fw_name
                    all_framework_dfs.append(df)
                    visualize_config_comparison(df, os.path.join(framework_dir, "config_comparison.html"))
                except Exception as e: print(f"  Error reading {csv_path}: {e}")

            stats_files = glob.glob(os.path.join(framework_dir, "stats_*.json"))
            for s_path in stats_files:
                try:
                    with open(s_path, 'r') as f:
                        data = json.load(f)
                        if data:
                            sdf = pd.DataFrame(data)
                            sdf['framework'] = fw_name
                            sdf['config_name'] = os.path.basename(s_path).replace('stats_', '').replace('.json', '')
                            all_framework_stats.append(sdf)
                except: pass

        if all_framework_dfs:
            combined = pd.concat(all_framework_dfs, ignore_index=True)
            visualize_framework_comparison(combined, pipeline_dir)
            
        if all_framework_stats:
            combined_stats = pd.concat(all_framework_stats, ignore_index=True)
            reconcile_stats(combined_stats, pipeline_name)

    print("\n--- Summarization Complete ---")