import duckdb
import pandas as pd
import polars as pl
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession, Window
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from utils import (
    timeit,
    duckdb_load_data,
    pandas_load_data,
    polars_load_data,
    scikit_load_data,
    spark_load_data,
    TRAIN_IDS_PARQUET,
    TEST_IDS_PARQUET
)

# =============================================================================
# --- 1. DUCKDB BENCHMARK ---
# =============================================================================

@timeit
def duckdb_create_macros(duckdb_conn):
    duckdb_conn.sql("""
        CREATE OR REPLACE MACRO scaling_params(table_name, column_list) AS TABLE
        FROM query_table(table_name)
        SELECT
            avg(columns(column_list)) as 'avg_\\0',
            stddev_pop(columns(column_list)) as 'std_\\0',
            min(columns(column_list)) as 'min_\\0',
            max(columns(column_list)) as 'max_\\0',
            quantile_cont(columns(column_list), 0.25) AS 'q25_\\0',
            quantile_cont(columns(column_list), 0.50) AS 'q50_\\0',
            quantile_cont(columns(column_list), 0.75) AS 'q75_\\0',
            median(columns(column_list)) as 'median_\\0';    
    """)
    duckdb_conn.sql("""
    CREATE OR REPLACE MACRO standard_scaler(val, avg_val, std_val) AS
        (val - avg_val)/std_val;
    """)
    duckdb_conn.sql("""
    CREATE OR REPLACE MACRO min_max_scaler(val, min_val, max_val) AS
        (val - min_val)/nullif(max_val - min_val, 0);
    """)
    duckdb_conn.sql("""
    CREATE OR REPLACE MACRO robust_scaler(val, q25_val, q50_val, q75_val) AS
        (val - q50_val)/nullif(q75_val - q25_val, 0);
    """)

@timeit
def duckdb_encode(duckdb_conn):
    duckdb_conn.sql("DROP TABLE IF EXISTS financial_trx_encoded")
    duckdb_conn.sql("""
        CREATE TABLE financial_trx_encoded AS
        WITH
            onehot_payment_channel AS (
                PIVOT financial_trx
                ON payment_channel
                USING COALESCE (MAX (payment_channel=payment_channel):: INT,0) AS 'onehot'
                GROUP BY payment_channel
            ),
            onehot_device_used AS (
                PIVOT financial_trx
                ON device_used
                USING COALESCE (MAX (device_used=device_used):: INT,0) AS 'onehot'
                GROUP BY device_used
            ),
            onehot_merchant_category AS (
                PIVOT financial_trx
                ON merchant_category
                USING COALESCE (MAX (merchant_category=merchant_category):: INT,0) AS 'onehot'
                GROUP BY merchant_category
            ),
            trx_type_ordinal_encoded AS (
                SELECT
                    transaction_type,
                    row_number() over (order by transaction_type) - 1 AS ordinal__transaction_type
                FROM (
                    SELECT DISTINCT transaction_type
                    FROM financial_trx
                    )
            )
            SELECT *
            FROM financial_trx trx
        JOIN onehot_payment_channel USING (payment_channel)
        JOIN onehot_device_used USING (device_used)
        JOIN onehot_merchant_category USING (merchant_category)
        JOIN trx_type_ordinal_encoded USING (transaction_type)
    """)

@timeit
def duckdb_split_data(duckdb_conn):
    duckdb_conn.sql(f"CREATE OR REPLACE TABLE train_ids AS FROM '{TRAIN_IDS_PARQUET}'")
    duckdb_conn.sql(f"CREATE OR REPLACE TABLE test_ids AS FROM '{TEST_IDS_PARQUET}'")
    
    duckdb_conn.sql("DROP TABLE IF EXISTS financial_trx_training")
    duckdb_conn.sql("""
        CREATE TABLE financial_trx_training AS
        SELECT f.*
        FROM financial_trx_encoded AS f
        JOIN train_ids USING (transaction_id)
    """)
    
    duckdb_conn.sql("DROP TABLE IF EXISTS financial_trx_testing")
    duckdb_conn.sql("""
        CREATE TABLE financial_trx_testing AS
        SELECT f.*
        FROM financial_trx_encoded AS f
        JOIN test_ids USING (transaction_id)
    """)

@timeit
def duckdb_feature_scaling_training_data(duckdb_conn):
    return duckdb_conn.sql("""
        SELECT
            transaction_id,
            * LIKE '%onehot',
            ordinal__transaction_type,
            ss_velocity_score: standard_scaler(
                velocity_score,
                avg_velocity_score,
                std_velocity_score
            ),
            min_max_spending_deviation_score: min_max_scaler(
                spending_deviation_score,
                min_spending_deviation_score,
                max_spending_deviation_score
            ),
            min_max_time_since_last_transaction : min_max_scaler(
                coalesce(time_since_last_transaction, avg_time_since_last_transaction),
                min_time_since_last_transaction,
                max_time_since_last_transaction
            ),
            rs_amount: robust_scaler(
                amount,
                q25_amount,
                q50_amount,
                q75_amount
            )
        FROM financial_trx_training,
             scaling_params(
                'financial_trx_training',
                ['velocity_score', 'spending_deviation_score', 'amount', 'time_since_last_transaction']
             )
    """).to_df()

@timeit
def duckdb_feature_scaling_testing_data(duckdb_conn):
    return duckdb_conn.sql("""
        SELECT
            transaction_id,
            * LIKE '%onehot',
            ordinal__transaction_type,
            ss_velocity_score: standard_scaler(
                velocity_score,
                avg_velocity_score,
                std_velocity_score
            ),
            min_max_spending_deviation_score: min_max_scaler(
                spending_deviation_score,
                min_spending_deviation_score,
                max_spending_deviation_score
            ),
            min_max_time_since_last_transaction : min_max_scaler(
                coalesce(time_since_last_transaction, avg_time_since_last_transaction),
                min_time_since_last_transaction,
                max_time_since_last_transaction
            ),
            rs_amount: robust_scaler(
                amount,
                q25_amount,
                q50_amount,
                q75_amount
            )
        FROM financial_trx_testing,
             scaling_params(
                'financial_trx_training', -- Use training params
                ['velocity_score', 'spending_deviation_score', 'amount', 'time_since_last_transaction']
             )
    """).to_df()

def run_duckdb_benchmark():
    """Runs the full DuckDB pipeline."""
    conn = duckdb.connect()
    duckdb_load_data(conn)
    duckdb_create_macros(conn)
    duckdb_encode(conn)
    duckdb_split_data(conn)
    x_train_transformed_df = duckdb_feature_scaling_training_data(conn)
    x_test_transformed_df = duckdb_feature_scaling_testing_data(conn)
    conn.close()
    return x_train_transformed_df, x_test_transformed_df


# =============================================================================
# --- 2. PANDAS BENCHMARK ---
# =============================================================================

@timeit
def pandas_encode(financial_trx):
    financial_trx_encoded = pd.get_dummies(
        financial_trx, 
        columns=['payment_channel', 'device_used', 'merchant_category'],
        prefix=['onehot_payment_channel', 'onehot_device_used', 'onehot_merchant_category'],
        dtype=int
    )
    financial_trx_encoded['ordinal__transaction_type'] = (
        financial_trx_encoded['transaction_type']
        .astype('category')
        .cat.codes
    )
    return financial_trx_encoded

@timeit
def pandas_split_data(financial_trx_encoded):
    train_ids = pd.read_parquet(TRAIN_IDS_PARQUET)
    test_ids = pd.read_parquet(TEST_IDS_PARQUET)
    
    x_train_df = financial_trx_encoded.merge(train_ids, on='transaction_id', how='inner')
    x_test_df = financial_trx_encoded.merge(test_ids, on='transaction_id', how='inner')
    return x_train_df, x_test_df

def get_pandas_scaling_params(x_train_df):
    params = {}
    params['avg_velocity_score'] = x_train_df['velocity_score'].mean()
    params['std_velocity_score'] = x_train_df['velocity_score'].std(ddof=0)
    params['min_spending_deviation_score'] = x_train_df['spending_deviation_score'].min()
    params['max_spending_deviation_score'] = x_train_df['spending_deviation_score'].max()
    params['avg_time_since_last_transaction'] = x_train_df['time_since_last_transaction'].mean()
    params['min_time_since_last_transaction'] = x_train_df['time_since_last_transaction'].min()
    params['max_time_since_last_transaction'] = x_train_df['time_since_last_transaction'].max()
    params['q25_amount'] = x_train_df['amount'].quantile(0.25)
    params['q50_amount'] = x_train_df['amount'].quantile(0.50)
    params['q75_amount'] = x_train_df['amount'].quantile(0.75)
    return params

def apply_pandas_scaling(df, params):
    df_scaled = df.copy()
    
    df_scaled['time_since_last_transaction'] = (
        df_scaled['time_since_last_transaction']
        .fillna(params['avg_time_since_last_transaction'])
    )
    
    df_scaled['ss_velocity_score'] = (
        (df_scaled['velocity_score'] - params['avg_velocity_score']) / 
        params['std_velocity_score']
    )
    
    min_max_denom_spend = params['max_spending_deviation_score'] - params['min_spending_deviation_score']
    df_scaled['min_max_spending_deviation_score'] = (
        (df_scaled['spending_deviation_score'] - params['min_spending_deviation_score']) /
        (min_max_denom_spend if min_max_denom_spend != 0 else pd.NA)
    )
    min_max_denom_time = params['max_time_since_last_transaction'] - params['min_time_since_last_transaction']
    df_scaled['min_max_time_since_last_transaction'] = (
        (df_scaled['time_since_last_transaction'] - params['min_time_since_last_transaction']) /
        (min_max_denom_time if min_max_denom_time != 0 else pd.NA)
    )
    
    robust_denom_amount = params['q75_amount'] - params['q25_amount']
    df_scaled['rs_amount'] = (
        (df_scaled['amount'] - params['q50_amount']) /
        (robust_denom_amount if robust_denom_amount != 0 else pd.NA)
    )
    
    one_hot_cols = [col for col in df_scaled.columns if 'onehot_' in col]
    final_cols = (
        ['transaction_id'] + 
        one_hot_cols + 
        ['ordinal__transaction_type', 'ss_velocity_score', 
         'min_max_spending_deviation_score', 'min_max_time_since_last_transaction', 'rs_amount']
    )
    return df_scaled[final_cols]

@timeit
def pandas_feature_scaling_training_data(x_train_df, scaling_params):
    return apply_pandas_scaling(x_train_df, scaling_params)

@timeit
def pandas_feature_scaling_testing_data(x_test_df, scaling_params):
    return apply_pandas_scaling(x_test_df, scaling_params)

def run_pandas_benchmark():
    """Runs the full Pandas pipeline."""
    raw_data_df = pandas_load_data()
    encoded_df = pandas_encode(raw_data_df)
    x_train_df, x_test_df = pandas_split_data(encoded_df)
    
    scaling_params = get_pandas_scaling_params(x_train_df) 
    
    x_train_transformed_df = pandas_feature_scaling_training_data(x_train_df, scaling_params)
    x_test_transformed_df = pandas_feature_scaling_testing_data(x_test_df, scaling_params)
    return x_train_transformed_df, x_test_transformed_df


# =============================================================================
# --- 3. POLARS BENCHMARK ---
# =============================================================================

def standard_scaler(val, avg_val, std_val):
    return (val - avg_val) / std_val

def min_max_scaler(val, min_val, max_val):
    denom = max_val - min_val
    return (val - min_val) / denom.replace(0, None) 

def robust_scaler(val, q25_val, q50_val, q75_val):
    denom = q75_val - q25_val
    return (val - q50_val) / denom.replace(0, None)

@timeit
def polars_encode(financial_trx):
    """Encodes categorical features for the Polars DataFrame."""
    
    financial_trx_encoded = financial_trx.to_dummies(
        columns=['payment_channel', 'device_used', 'merchant_category']
    )
    
    rename_map = {
        col: f"onehot_payment_channel_{col.split('_', 2)[-1]}" for col in financial_trx_encoded.columns if col.startswith('payment_channel_')
    }
    rename_map.update({
        col: f"onehot_device_used_{col.split('_', 2)[-1]}" for col in financial_trx_encoded.columns if col.startswith('device_used_')
    })
    rename_map.update({
        col: f"onehot_merchant_category_{col.split('_', 2)[-1]}" for col in financial_trx_encoded.columns if col.startswith('merchant_category_')
    })
    financial_trx_encoded = financial_trx_encoded.rename(rename_map)

    trx_type_map = (
        financial_trx.select("transaction_type")
        .unique()
        .sort("transaction_type")
        .with_row_count(name="ordinal__transaction_type")
    )
    
    financial_trx_encoded = financial_trx_encoded.join(
        trx_type_map, on="transaction_type"
    )
    
    return financial_trx_encoded

@timeit
def polars_split_data(financial_trx_encoded):
    train_ids = pl.read_parquet(TRAIN_IDS_PARQUET)
    test_ids = pl.read_parquet(TEST_IDS_PARQUET)
    
    x_train_df = financial_trx_encoded.join(train_ids, on='transaction_id', how='inner')
    x_test_df = financial_trx_encoded.join(test_ids, on='transaction_id', how='inner')
    return x_train_df, x_test_df

def get_polars_scaling_params(x_train_df):
    scaling_params_df = x_train_df.select(
        pl.col('velocity_score').mean().alias('avg_velocity_score'),
        pl.col('velocity_score').std(ddof=0).alias('std_velocity_score'),
        pl.col('spending_deviation_score').min().alias('min_spending_deviation_score'),
        pl.col('spending_deviation_score').max().alias('max_spending_deviation_score'),
        pl.col('time_since_last_transaction').mean().alias('avg_time_since_last_transaction'),
        pl.col('time_since_last_transaction').min().alias('min_time_since_last_transaction'),
        pl.col('time_since_last_transaction').max().alias('max_time_since_last_transaction'),
        pl.col('amount').quantile(0.25).alias('q25_amount'),
        pl.col('amount').quantile(0.50).alias('q50_amount'),
        pl.col('amount').quantile(0.75).alias('q75_amount')
    )
    return scaling_params_df

def apply_polars_scaling(df, scaling_params_df):
    df_with_params = df.join(scaling_params_df, how='cross')
    
    df_scaled = df_with_params.with_columns(
        pl.col('time_since_last_transaction').fill_null(
            pl.col('avg_time_since_last_transaction')
        ).alias('imputed_time_since_last_transaction'),
    ).with_columns(
        ss_velocity_score=standard_scaler(
            pl.col('velocity_score'),
            pl.col('avg_velocity_score'),
            pl.col('std_velocity_score')
        ),
        min_max_spending_deviation_score=min_max_scaler(
            pl.col('spending_deviation_score'),
            pl.col('min_spending_deviation_score'),
            pl.col('max_spending_deviation_score')
        ),
        min_max_time_since_last_transaction=min_max_scaler(
            pl.col('imputed_time_since_last_transaction'),
            pl.col('min_time_since_last_transaction'),
            pl.col('max_time_since_last_transaction')
        ),
        rs_amount=robust_scaler(
            pl.col('amount'),
            pl.col('q25_amount'),
            pl.col('q50_amount'),
            pl.col('q75_amount')
        )
    )
    
    one_hot_cols = [col for col in df_scaled.columns if col.startswith('onehot_')]
    final_cols = (
        ['transaction_id'] + 
        one_hot_cols + 
        ['ordinal__transaction_type', 'ss_velocity_score', 
         'min_max_spending_deviation_score', 'min_max_time_since_last_transaction', 'rs_amount']
    )
    return df_scaled.select(final_cols)

@timeit
def polars_feature_scaling_training_data(x_train_df, scaling_params_df):
    return apply_polars_scaling(x_train_df, scaling_params_df)

@timeit
def polars_feature_scaling_testing_data(x_test_df, scaling_params_df):
    return apply_polars_scaling(x_test_df, scaling_params_df)

def run_polars_benchmark():
    """Runs the full Polars pipeline."""
    raw_data_df = polars_load_data()
    encoded_df = polars_encode(raw_data_df)
    x_train_df, x_test_df = polars_split_data(encoded_df)
    
    scaling_params_df = get_polars_scaling_params(x_train_df) 
    
    x_train_transformed_df = polars_feature_scaling_training_data(x_train_df, scaling_params_df)
    x_test_transformed_df = polars_feature_scaling_testing_data(x_test_df, scaling_params_df)
    
    return x_train_transformed_df, x_test_transformed_df


# =============================================================================
# --- 4. SCIKIT-LEARN BENCHMARK ---
# =============================================================================

@timeit
def scikit_encode(data_df):
    encoding_steps = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown='ignore'),
                ["payment_channel", "device_used", "merchant_category"],
            ),
            ("ordinal", OrdinalEncoder(), ["transaction_type"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    
    encoded_data = encoding_steps.fit_transform(data_df)

    df = pd.DataFrame(
        encoded_data,
        columns=encoding_steps.get_feature_names_out(),
        index=data_df.index,
    )
    return df

@timeit
def scikit_split_data(encoded_data_df):
    train_ids = pd.read_parquet(TRAIN_IDS_PARQUET)
    test_ids = pd.read_parquet(TEST_IDS_PARQUET)
    
    x_train_df = encoded_data_df.merge(train_ids, on='transaction_id', how='inner')
    x_test_df = encoded_data_df.merge(test_ids, on='transaction_id', how='inner')
    return x_train_df, x_test_df

@timeit
def scikit_feature_scaling_training_data(x_train):
    impute_missing_data = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler(copy=False)),
        ]
    )

    scaling_steps = ColumnTransformer(
        [
            ("ss", StandardScaler(copy=False), ["velocity_score"]),
            (
                "minmax_time_since_last_transaction",
                impute_missing_data,
                ["time_since_last_transaction"],
            ),
            ("minmax", MinMaxScaler(copy=False), ["spending_deviation_score"]),
            ("rs", RobustScaler(copy=False), ["amount"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    scaling_steps.set_output(transform="pandas")
    scaling_steps.fit(x_train)

    return scaling_steps, scaling_steps.transform(x_train)

@timeit
def scikit_feature_scaling_testing_data(scaling_steps, x_test):
    return scaling_steps.transform(x_test)

def run_scikit_benchmark():
    """Runs the full Scikit-learn pipeline."""
    raw_data_df = scikit_load_data()
    encoded_df = scikit_encode(raw_data_df) 
    x_train_df, x_test_df = scikit_split_data(encoded_df)
    scaling_steps_proc, x_train_transformed_df = (
        scikit_feature_scaling_training_data(x_train_df)
    )
    x_test_transformed_df = scikit_feature_scaling_testing_data(
        scaling_steps_proc, x_test_df
    )
    return x_train_transformed_df, x_test_transformed_df


# =============================================================================
# --- 5. SPARK BENCHMARK ---
# =============================================================================

def spark_standard_scaler(val, avg_val, std_val):
    return (val - avg_val) / std_val

def spark_min_max_scaler(val, min_val, max_val):
    denom = max_val - min_val
    return (val - min_val) / F.when(denom == 0, None).otherwise(denom)

def spark_robust_scaler(val, q25_val, q50_val, q75_val):
    denom = q75_val - q25_val
    return (val - q50_val) / F.when(denom == 0, None).otherwise(denom)

def create_onehot_lookup(df, col_name, prefix):
    distinct_vals = df.select(col_name).distinct()
    lookup_table = (
        distinct_vals
        .groupBy(col_name)
        .pivot(col_name)
        .agg(F.lit(1))
        .fillna(0)
    )
    for old_col_name in lookup_table.columns:
        if old_col_name != col_name:
            lookup_table = lookup_table.withColumnRenamed(
                old_col_name, f"{prefix}_{col_name}_{old_col_name}"
            )
    return lookup_table

@timeit
def spark_encode(financial_trx):
    """Encodes categorical features for the Spark DataFrame."""
    
    onehot_payment_channel = create_onehot_lookup(financial_trx, "payment_channel", "onehot")
    onehot_device_used = create_onehot_lookup(financial_trx, "device_used", "onehot")
    onehot_merchant_category = create_onehot_lookup(financial_trx, "merchant_category", "onehot")
    
    indexer = StringIndexer(
        inputCol="transaction_type",
        outputCol="ordinal__transaction_type",
        stringOrderType="alphabetAsc"
    )
    
    indexer_model = indexer.fit(financial_trx)
    financial_trx_with_ordinal = indexer_model.transform(financial_trx)
    
    financial_trx_encoded = (
        financial_trx_with_ordinal
        .join(onehot_payment_channel, "payment_channel")
        .join(onehot_device_used, "device_used")
        .join(onehot_merchant_category, "merchant_category")
    )
    
    return financial_trx_encoded

@timeit
def spark_split_data(financial_trx_encoded):
    spark = financial_trx_encoded.sparkSession
    train_ids = spark.read.parquet(TRAIN_IDS_PARQUET)
    test_ids = spark.read.parquet(TEST_IDS_PARQUET)
    
    x_train_df = financial_trx_encoded.join(
        train_ids, on='transaction_id', how='inner'
    )
    x_test_df = financial_trx_encoded.join(
        test_ids, on='transaction_id', how='inner'
    )
    return x_train_df, x_test_df

def get_spark_scaling_params(x_train_df):
    """Calculates scaling parameters into a 1-row DataFrame."""
    
    # 1. Get exact quantiles for 'amount'
    # This is an action and returns a list: [q25, q50, q75]
    # We use relativeError=0.0 to force an exact quantile calculation
    quantiles = x_train_df.approxQuantile("amount", [0.25, 0.5, 0.75], 0.0)
    
    # Get the other stats in a 1-row DataFrame
    other_stats_df = x_train_df.select(
        F.avg('velocity_score').alias('avg_velocity_score'),
        F.stddev_pop('velocity_score').alias('std_velocity_score'),
        
        F.min('spending_deviation_score').alias('min_spending_deviation_score'),
        F.max('spending_deviation_score').alias('max_spending_deviation_score'),
        
        F.avg('time_since_last_transaction').alias('avg_time_since_last_transaction'),
        F.min('time_since_last_transaction').alias('min_time_since_last_transaction'),
        F.max('time_since_last_transaction').alias('max_time_since_last_transaction')
    )
    
    # Add the exact quantiles to the 1-row DataFrame
    scaling_params_df = other_stats_df.withColumns({
        'q25_amount': F.lit(quantiles[0]),
        'q50_amount': F.lit(quantiles[1]),
        'q75_amount': F.lit(quantiles[2])
    })

    return scaling_params_df.cache() # Cache for reuse

def apply_spark_scaling(df, scaling_params_df):
    df_with_params = df.crossJoin(F.broadcast(scaling_params_df))
    
    df_scaled = df_with_params.withColumn(
        'imputed_time_since_last_transaction',
        F.coalesce(
            F.col('time_since_last_transaction'),
            F.col('avg_time_since_last_transaction')
        )
    ).withColumn(
        'ss_velocity_score',
        spark_standard_scaler(
            F.col('velocity_score'),
            F.col('avg_velocity_score'),
            F.col('std_velocity_score')
        )
    ).withColumn(
        'min_max_spending_deviation_score',
        spark_min_max_scaler(
            F.col('spending_deviation_score'),
            F.col('min_spending_deviation_score'),
            F.col('max_spending_deviation_score')
        )
    ).withColumn(
        'min_max_time_since_last_transaction',
        spark_min_max_scaler(
            F.col('imputed_time_since_last_transaction'),
            F.col('min_time_since_last_transaction'),
            F.col('max_time_since_last_transaction')
        )
    ).withColumn(
        'rs_amount',
        spark_robust_scaler(
            F.col('amount'),
            F.col('q25_amount'),
            F.col('q50_amount'),
            F.col('q75_amount')
        )
    )
    
    one_hot_cols = [col for col in df_scaled.columns if col.startswith('onehot_')]
    final_cols = (
        ['transaction_id'] + 
        one_hot_cols + 
        ['ordinal__transaction_type', 'ss_velocity_score', 
         'min_max_spending_deviation_score', 'min_max_time_since_last_transaction', 'rs_amount']
    )
    return df_scaled.select(final_cols)


@timeit
def spark_feature_scaling_training_data(x_train_df, scaling_params_df):
    df = apply_spark_scaling(x_train_df, scaling_params_df)
    return df.toPandas()

@timeit
def spark_feature_scaling_testing_data(x_test_df, scaling_params_df):
    df = apply_spark_scaling(x_test_df, scaling_params_df)
    return df.toPandas()

def run_spark_benchmark(spark: SparkSession):
    """Runs the full Spark pipeline."""
    raw_data_df = spark_load_data(spark)
    encoded_df = spark_encode(raw_data_df)
    x_train_df, x_test_df = spark_split_data(encoded_df)
    
    scaling_params_df = get_spark_scaling_params(x_train_df)
    
    x_train_transformed_df = spark_feature_scaling_training_data(x_train_df, scaling_params_df)
    x_test_transformed_df = spark_feature_scaling_testing_data(x_test_df, scaling_params_df)
    
    return x_train_transformed_df, x_test_transformed_df