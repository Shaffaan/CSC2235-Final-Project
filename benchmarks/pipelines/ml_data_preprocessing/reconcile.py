import duckdb
import pandas as pd
import polars as pl

# Define the canonical column structure for comparison
CANONICAL_COLUMNS = [
    'transaction_id',
    'onehot_payment_channel_ACH',
    'onehot_payment_channel_UPI',
    'onehot_payment_channel_card',
    'onehot_payment_channel_wire_transfer',
    'onehot_device_used_atm',
    'onehot_device_used_mobile',
    'onehot_device_used_pos',
    'onehot_device_used_web',
    'onehot_merchant_category_entertainment',
    'onehot_merchant_category_grocery',
    'onehot_merchant_category_online',
    'onehot_merchant_category_other',
    'onehot_merchant_category_restaurant',
    'onehot_merchant_category_retail',
    'onehot_merchant_category_travel',
    'onehot_merchant_category_utilities',
    'ordinal__transaction_type',
    'ss_velocity_score',
    'min_max_spending_deviation_score',
    'min_max_time_since_last_transaction',
    'rs_amount'
]

def _get_select_statement(library: str) -> str:
    """
    Generates a SQL SELECT statement to standardize column names
    for each library's output.
    """
    if library == 'duckdb':
        return """
        SELECT
            transaction_id,
            ACH_onehot AS onehot_payment_channel_ACH,
            UPI_onehot AS onehot_payment_channel_UPI,
            card_onehot AS onehot_payment_channel_card,
            wire_transfer_onehot AS onehot_payment_channel_wire_transfer,
            atm_onehot AS onehot_device_used_atm,
            mobile_onehot AS onehot_device_used_mobile,
            pos_onehot AS onehot_device_used_pos,
            web_onehot AS onehot_device_used_web,
            entertainment_onehot AS onehot_merchant_category_entertainment,
            grocery_onehot AS onehot_merchant_category_grocery,
            online_onehot AS onehot_merchant_category_online,
            other_onehot AS onehot_merchant_category_other,
            restaurant_onehot AS onehot_merchant_category_restaurant,
            retail_onehot AS onehot_merchant_category_retail,
            travel_onehot AS onehot_merchant_category_travel,
            utilities_onehot AS onehot_merchant_category_utilities,
            ordinal__transaction_type,
            round(ss_velocity_score, 8) AS ss_velocity_score,
            round(min_max_spending_deviation_score, 8) AS min_max_spending_deviation_score,
            round(min_max_time_since_last_transaction, 8) AS min_max_time_since_last_transaction,
            round(rs_amount, 8) AS rs_amount
        FROM {table_name}
        """
    elif library == 'scikit':
        return """
        SELECT
            transaction_id,
            payment_channel_ACH AS onehot_payment_channel_ACH,
            payment_channel_UPI AS onehot_payment_channel_UPI,
            payment_channel_card AS onehot_payment_channel_card,
            payment_channel_wire_transfer AS onehot_payment_channel_wire_transfer,
            device_used_atm AS onehot_device_used_atm,
            device_used_mobile AS onehot_device_used_mobile,
            device_used_pos AS onehot_device_used_pos,
            device_used_web AS onehot_device_used_web,
            merchant_category_entertainment AS onehot_merchant_category_entertainment,
            merchant_category_grocery AS onehot_merchant_category_grocery,
            merchant_category_online AS onehot_merchant_category_online,
            merchant_category_other AS onehot_merchant_category_other,
            merchant_category_restaurant AS onehot_merchant_category_restaurant,
            merchant_category_retail AS onehot_merchant_category_retail,
            merchant_category_travel AS onehot_merchant_category_travel,
            merchant_category_utilities AS onehot_merchant_category_utilities,
            CAST(transaction_type AS BIGINT) AS ordinal__transaction_type,
            round(velocity_score, 8) AS ss_velocity_score,
            round(spending_deviation_score, 8) AS min_max_spending_deviation_score,
            round(time_since_last_transaction, 8) AS min_max_time_since_last_transaction,
            round(amount, 8) AS rs_amount
        FROM {table_name}
        """
    else: # pandas, polars, spark
        return """
        SELECT
            transaction_id,
            onehot_payment_channel_ACH,
            onehot_payment_channel_UPI,
            onehot_payment_channel_card,
            onehot_payment_channel_wire_transfer,
            onehot_device_used_atm,
            onehot_device_used_mobile,
            onehot_device_used_pos,
            onehot_device_used_web,
            onehot_merchant_category_entertainment,
            onehot_merchant_category_grocery,
            onehot_merchant_category_online,
            onehot_merchant_category_other,
            onehot_merchant_category_restaurant,
            onehot_merchant_category_retail,
            onehot_merchant_category_travel,
            onehot_merchant_category_utilities,
            CAST(ordinal__transaction_type AS BIGINT) AS ordinal__transaction_type,
            round(ss_velocity_score, 8) AS ss_velocity_score,
            round(min_max_spending_deviation_score, 8) AS min_max_spending_deviation_score,
            round(min_max_time_since_last_transaction, 8) AS min_max_time_since_last_transaction,
            round(rs_amount, 8) AS rs_amount
        FROM {table_name}
        """

def reconcile_all_results(results_dict: dict):
    """
    Compares the outputs of all pipelines against a base (DuckDB).
    """
    conn = duckdb.connect()
    
    base_lib = 'duckdb'
    
    # Check if base library data exists
    if base_lib not in results_dict:
        print(f"[FAIL] Base library '{base_lib}' data not found. Skipping reconciliation.")
        return False
        
    base_train_df, base_test_df = results_dict[base_lib]
    
    # Register the base tables
    conn.register('base_train_tbl', base_train_df)
    conn.register('base_test_tbl', base_test_df)
    
    base_train_sql = _get_select_statement(base_lib).format(table_name='base_train_tbl')
    base_test_sql = _get_select_statement(base_lib).format(table_name='base_test_tbl')

    all_passed = True

    for library, data in results_dict.items():
        if library == base_lib:
            continue
            
        # Check if library data exists
        if data is None:
            print(f"--- Reconciling vs. {library.upper()} ---")
            print(f"  [FAIL] Data for '{library}' is missing (benchmark may have failed).")
            all_passed = False
            continue
            
        train_df, test_df = data
        
        print(f"--- Reconciling vs. {library.upper()} ---")
        
        # Convert Polars to Pandas for DuckDB registration
        if isinstance(train_df, pl.DataFrame):
            train_df = train_df.to_pandas()
        if isinstance(test_df, pl.DataFrame):
            test_df = test_df.to_pandas()

        # Register the comparison tables
        conn.register(f'{library}_train_tbl', train_df)
        conn.register(f'{library}_test_tbl', test_df)
        
        lib_train_sql = _get_select_statement(library).format(table_name=f'{library}_train_tbl')
        lib_test_sql = _get_select_statement(library).format(table_name=f'{library}_test_tbl')

        try:
            query_base_except_lib = f"({base_train_sql}) EXCEPT ({lib_train_sql})"
            diff_count_train = conn.sql(f"SELECT COUNT(*) FROM ({query_base_except_lib})").fetchone()[0]
            
            query_lib_except_base = f"({lib_train_sql}) EXCEPT ({base_train_sql})"
            diff_count_train += conn.sql(f"SELECT COUNT(*) FROM ({query_lib_except_base})").fetchone()[0]

            if diff_count_train == 0:
                print(f"  [PASS] Training data matches {base_lib}.")
            else:
                print(f"  [FAIL] Training data has {diff_count_train} mismatched rows vs. {base_lib}.")
                all_passed = False

            query_base_except_lib = f"({base_test_sql}) EXCEPT ({lib_test_sql})"
            diff_count_test = conn.sql(f"SELECT COUNT(*) FROM ({query_base_except_lib})").fetchone()[0]
            
            query_lib_except_base = f"({lib_test_sql}) EXCEPT ({base_test_sql})"
            diff_count_test += conn.sql(f"SELECT COUNT(*) FROM ({query_lib_except_base})").fetchone()[0]
            
            if diff_count_test == 0:
                print(f"  [PASS] Testing data matches {base_lib}.")
            else:
                print(f"  [FAIL] Testing data has {diff_count_test} mismatched rows vs. {base_lib}.")
                all_passed = False
        
        except Exception as e:
            print(f"  [FAIL] Error during '{library}' reconciliation: {e}")
            all_passed = False
            
    return all_passed