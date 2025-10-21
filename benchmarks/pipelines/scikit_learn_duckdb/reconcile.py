import duckdb
import pandas as pd
import polars as pl

def reconcile_all_results(results_dict: dict):
    """
    Compares the outputs of all ML pipelines.
    1. Checks if all accuracies are identical.
    2. Checks if all prediction DataFrames are identical.
    """
    
    # --- 1. Reconcile Accuracy ---
    print("--- Reconciling Model Accuracy ---")
    base_accuracy = None
    all_acc_passed = True
    
    for library, results in results_dict.items():
        if "accuracy" not in results:
            print(f"  [FAIL] '{library}' is missing 'accuracy' result.")
            all_acc_passed = False
            continue
            
        acc = results["accuracy"]
        if base_accuracy is None:
            base_accuracy = acc
            print(f"  [BASE] {library} Accuracy: {acc}")
        elif acc != base_accuracy:
            print(f"  [FAIL] {library} Accuracy ({acc}) does not match base ({base_accuracy}).")
            all_acc_passed = False
        else:
            print(f"  [PASS] {library} Accuracy matches base.")
            
    if not all_acc_passed:
        print("Accuracy reconciliation FAILED.")
    else:
        print("Accuracy reconciliation PASSED.")

    # --- 2. Reconcile Predictions ---
    print("\n--- Reconciling Predictions ---")
    conn = duckdb.connect()
    base_lib = 'duckdb'
    
    if base_lib not in results_dict or "predictions" not in results_dict[base_lib]:
        print(f"[FAIL] Base library '{base_lib}' prediction data not found.")
        return False
        
    base_pred_df = results_dict[base_lib]["predictions"]
    conn.register('base_tbl', base_pred_df)
    
    base_sql = "SELECT * FROM base_tbl ORDER BY observation_id"
    all_pred_passed = True

    for library, results in results_dict.items():
        if library == base_lib:
            continue
            
        if "predictions" not in results:
            print(f"  [FAIL] '{library}' is missing 'predictions' result.")
            all_pred_passed = False
            continue
            
        print(f"--- Reconciling vs. {library.upper()} ---")
        pred_df = results["predictions"]
        
        # Convert Polars to Pandas for registration
        if isinstance(pred_df, pl.DataFrame):
            pred_df = pred_df.to_pandas()
            
        conn.register(f'{library}_tbl', pred_df)
        lib_sql = f"SELECT * FROM {library}_tbl ORDER BY observation_id"
        
        try:
            query1 = f"({base_sql}) EXCEPT ({lib_sql})"
            diff_count = conn.sql(f"SELECT COUNT(*) FROM ({query1})").fetchone()[0]
            
            query2 = f"({lib_sql}) EXCEPT ({base_sql})"
            diff_count += conn.sql(f"SELECT COUNT(*) FROM ({query2})").fetchone()[0]

            if diff_count == 0:
                print(f"  [PASS] Predictions match {base_lib}.")
            else:
                print(f"  [FAIL] Predictions have {diff_count} mismatched rows vs. {base_lib}.")
                all_pred_passed = False
        
        except Exception as e:
            print(f"  [FAIL] Error during '{library}' reconciliation: {e}")
            all_pred_passed = False
            
    if not all_pred_passed:
        print("Prediction reconciliation FAILED.")
    else:
        print("Prediction reconciliation PASSED.")

    return all_acc_passed and all_pred_passed