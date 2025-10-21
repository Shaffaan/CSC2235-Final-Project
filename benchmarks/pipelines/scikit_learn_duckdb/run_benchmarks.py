import duckdb
import pandas as pd
import polars as pl
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

from utils import (
    timeit,
    LOCAL_CSV,
    TRAIN_IDS_PARQUET,
    TEST_IDS_PARQUET
)

# Define the features and model for all libraries
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "island_id"]
TARGET = "species_id"
MODEL = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=5)

# =============================================================================
# --- 1. DUCKDB BENCHMARK ---
# =============================================================================

@timeit
def duckdb_load_data(conn):
    """Loads and preprocesses data in DuckDB."""
    (
        conn.read_csv(LOCAL_CSV)
        .filter("columns(*)::text != 'NA'")
        .filter("columns(*) is not null")
        .select("*, row_number() over () as observation_id")
    ).to_table("penguins_data")

    conn.sql("alter table penguins_data alter bill_length_mm set data type decimal(5, 2)")
    conn.sql("alter table penguins_data alter bill_depth_mm  set data type decimal(5, 2)")
    conn.sql("alter table penguins_data alter body_mass_g set data type integer")
    conn.sql("alter table penguins_data alter flipper_length_mm set data type integer")

@timeit
def duckdb_encode_data(conn):
    """Creates ordinal encoded tables and joins them."""
    for feature in ["species", "island"]:
        conn.sql(f"DROP TABLE IF EXISTS {feature}_ref")
        (
            conn.table("penguins_data")
            .select(feature)
            .distinct()
            .order(feature)
            .select(f"{feature}, row_number() over () - 1 as {feature}_id")
        ).to_table(f"{feature}_ref")
    
    conn.sql("""
        CREATE OR REPLACE TABLE penguins_encoded AS
        SELECT p.*, s.species_id, i.island_id
        FROM penguins_data p
        JOIN species_ref s ON p.species = s.species
        JOIN island_ref i ON p.island = i.island
    """)

@timeit
def duckdb_split_data(conn):
    """Splits data using uniform IDs."""
    conn.sql(f"CREATE OR REPLACE TABLE train_ids AS FROM '{TRAIN_IDS_PARQUET}'")
    conn.sql(f"CREATE OR REPLACE TABLE test_ids AS FROM '{TEST_IDS_PARQUET}'")
    
    conn.sql("""
        CREATE OR REPLACE TABLE train_df AS
        SELECT * FROM penguins_encoded
        WHERE observation_id IN (SELECT observation_id FROM train_ids)
    """)
    conn.sql("""
        CREATE OR REPLACE TABLE test_df AS
        SELECT * FROM penguins_encoded
        WHERE observation_id IN (SELECT observation_id FROM test_ids)
    """)
    
    # Cast Decimal to Float for scikit-learn
    feature_select_sql = ", ".join([
        f"CAST({col} AS FLOAT) AS {col}" if "mm" in col else col 
        for col in FEATURES
    ])
    
    X_train = conn.table("train_df").select(feature_select_sql).to_df()
    y_train = conn.table("train_df").select(TARGET).df()[TARGET].values
    X_test = conn.table("test_df").select(feature_select_sql).to_df()
    y_test = conn.table("test_df").select(TARGET).df()[TARGET].values
    
    return X_train, X_test, y_train, y_test, conn.table("test_df"), conn

@timeit
def duckdb_train_model(data):
    """Trains the model on DuckDB-prepared data."""
    X_train, X_test, y_train, y_test, _, _ = data
    model = MODEL.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"  DuckDB Accuracy: {accuracy}")
    return model, accuracy

@timeit
def duckdb_predict_model(data):
    """Predicts using a DuckDB UDF, passing arrays directly."""
    X_train, X_test, y_train, y_test, test_df, conn = data
    model = MODEL.fit(X_train, y_train)

    def predict_batch_arrays(
        bill_length_mm: list[float],
        bill_depth_mm: list[float],
        flipper_length_mm: list[int],
        body_mass_g: list[int],
        island_id: list[int]
    ) -> list[int]:

        input_data_np = np.stack(
            [
                bill_length_mm,
                bill_depth_mm,
                flipper_length_mm,
                body_mass_g,
                island_id
            ],
            axis=1
        )
        return model.predict(input_data_np).tolist()

    try:
        conn.remove_function("predict_species_arrays")
    except Exception:
        pass

    conn.create_function(
        "predict_species_arrays",
        predict_batch_arrays,
        [
            'FLOAT[]',
            'FLOAT[]',
            'INTEGER[]',
            'INTEGER[]',
            'INTEGER[]'
        ],
        'INTEGER[]'
    )

    agg_query = f"""
        SELECT
            array_agg(CAST(bill_length_mm AS FLOAT)) as bill_lengths,
            array_agg(CAST(bill_depth_mm AS FLOAT)) as bill_depths,
            array_agg(flipper_length_mm) as flipper_lengths,
            array_agg(body_mass_g) as body_masses,
            array_agg(CAST(island_id AS INTEGER)) as island_ids, -- Cast to INTEGER
            array_agg(observation_id) as observation_ids,
            array_agg(species_id) as species_ids
        FROM test_df
    """

    predictions = conn.sql(f"""
        WITH aggregated_data AS ({agg_query})
        SELECT
           unnest(observation_ids) as observation_id,
           unnest(species_ids) as species_id,
           unnest(predict_species_arrays(
               bill_lengths,
               bill_depths,
               flipper_lengths,
               body_masses,
               island_ids -- Now correctly INTEGER[]
            )) as predicted_species_id
        FROM aggregated_data
    """).to_df()

    return predictions

def run_duckdb_benchmark():
    conn = duckdb.connect() 
    duckdb_load_data(conn)
    duckdb_encode_data(conn)
    split_data = duckdb_split_data(conn)
    _, accuracy = duckdb_train_model(split_data)
    predictions = duckdb_predict_model(split_data)
    conn.close()
    return {"accuracy": accuracy, "predictions": predictions}

# =============================================================================
# --- 2. PANDAS BENCHMARK ---
# =============================================================================

@timeit
def pandas_load_data():
    """Loads and preprocesses data in Pandas."""
    df = pd.read_csv(LOCAL_CSV)
    df = df.replace('NA', np.nan).dropna()
    df["observation_id"] = range(1, len(df) + 1)
    
    df["bill_length_mm"] = df["bill_length_mm"].astype(float)
    df["bill_depth_mm"] = df["bill_depth_mm"].astype(float)
    df["body_mass_g"] = df["body_mass_g"].astype(int)
    df["flipper_length_mm"] = df["flipper_length_mm"].astype(int)
    return df

@timeit
def pandas_encode_data(df):
    """Creates ordinal encoded columns."""
    # Pandas .cat.codes sorts alphabetically by default
    df["species_id"] = df["species"].astype("category").cat.codes
    df["island_id"] = df["island"].astype("category").cat.codes
    return df

@timeit
def pandas_split_data(df):
    """Splits data using uniform IDs."""
    train_ids = pd.read_parquet(TRAIN_IDS_PARQUET)
    test_ids = pd.read_parquet(TEST_IDS_PARQUET)
    
    train_df = df.merge(train_ids, on="observation_id", how="inner")
    test_df = df.merge(test_ids, on="observation_id", how="inner")
    
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values
    
    return X_train, X_test, y_train, y_test, test_df

@timeit
def pandas_train_model(data):
    """Trains the model on Pandas-prepared data."""
    X_train, X_test, y_train, y_test, _ = data
    model = MODEL.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"  Pandas Accuracy: {accuracy}")
    return model, accuracy

@timeit
def pandas_predict_model(data):
    """Predicts using model.predict()."""
    X_train, X_test, y_train, y_test, test_df = data
    model = MODEL.fit(X_train, y_train) # Re-fit to not time model loading

    predictions = model.predict(X_test)
    
    result_df = test_df[["observation_id", "species_id"]].copy()
    result_df["predicted_species_id"] = predictions
    return result_df

def run_pandas_benchmark():
    df = pandas_load_data()
    df_encoded = pandas_encode_data(df)
    split_data = pandas_split_data(df_encoded)
    _, accuracy = pandas_train_model(split_data)
    predictions = pandas_predict_model(split_data)
    return {"accuracy": accuracy, "predictions": predictions}

# =============================================================================
# --- 3. POLARS BENCHMARK ---
# =============================================================================

@timeit
def polars_load_data():
    """Loads and preprocesses data in Polars."""
    df = (
        pl.read_csv(LOCAL_CSV, null_values="NA")
        .drop_nulls() # Correct method name
        .with_row_count(name="observation_id", offset=1)
        .with_columns(
            pl.col("bill_length_mm").cast(pl.Float64),
            pl.col("bill_depth_mm").cast(pl.Float64),
            pl.col("body_mass_g").cast(pl.Int64),
            pl.col("flipper_length_mm").cast(pl.Int64),
        )
    )
    return df

@timeit
def polars_encode_data(df):
    """Creates ordinal encoded columns."""
    df_encoded = df
    for feature in ["species", "island"]:
        mapping_df = (
            df.select(feature)
            .unique()
            .sort(feature)
            .with_row_count(name=f"{feature}_id")
        )
        # Join mapping back to main df
        df_encoded = df_encoded.join(mapping_df, on=feature)
    return df_encoded

@timeit
def polars_split_data(df):
    """Splits data using uniform IDs."""
    train_ids = pl.read_parquet(TRAIN_IDS_PARQUET)
    test_ids = pl.read_parquet(TEST_IDS_PARQUET)
    
    train_df = df.join(train_ids, on="observation_id", how="inner")
    test_df = df.join(test_ids, on="observation_id", how="inner")
    
    X_train = train_df.select(FEATURES).to_numpy()
    y_train = train_df.select(TARGET).to_numpy().ravel()
    X_test = test_df.select(FEATURES).to_numpy()
    y_test = test_df.select(TARGET).to_numpy().ravel()
    
    return X_train, X_test, y_train, y_test, test_df

@timeit
def polars_train_model(data):
    """Trains the model on Polars-prepared data."""
    X_train, X_test, y_train, y_test, _ = data
    model = MODEL.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"  Polars Accuracy: {accuracy}")
    return model, accuracy

@timeit
def polars_predict_model(data):
    """Predicts using model.predict()."""
    X_train, X_test, y_train, y_test, test_df = data
    model = MODEL.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
    result_df = test_df.select(["observation_id", "species_id"]).with_columns(
        pl.lit(predictions).cast(pl.Int64).alias("predicted_species_id")
    )
    return result_df.to_pandas()

def run_polars_benchmark():
    df = polars_load_data()
    df_encoded = polars_encode_data(df)
    split_data = polars_split_data(df_encoded)
    _, accuracy = polars_train_model(split_data)
    predictions = polars_predict_model(split_data)
    return {"accuracy": accuracy, "predictions": predictions}

# =============================================================================
# --- 4. SCIKIT-LEARN BENCHMARK ---
# =============================================================================

@timeit
def scikit_load_data():
    """Loads data into Pandas for Scikit-learn."""
    df = pd.read_csv(LOCAL_CSV)
    df = df.replace('NA', np.nan).dropna()
    df["observation_id"] = range(1, len(df) + 1)
    
    # Keep numeric types as float for sklearn transformers
    df["bill_length_mm"] = df["bill_length_mm"].astype(float)
    df["bill_depth_mm"] = df["bill_depth_mm"].astype(float)
    df["body_mass_g"] = df["body_mass_g"].astype(float)
    df["flipper_length_mm"] = df["flipper_length_mm"].astype(float)
    return df

@timeit
def scikit_encode_data(df):
    """Encodes data using Scikit-learn's ColumnTransformer."""
    # Scikit's OrdinalEncoder sorts alphabetically by default
    encoder = ColumnTransformer(
        [
            ("species_enc", OrdinalEncoder(dtype=int), ["species"]),
            ("island_enc", OrdinalEncoder(dtype=int), ["island"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    
    encoded_data = encoder.fit_transform(df)
    
    # Reconstruct DataFrame
    feature_names = encoder.get_feature_names_out()
    df_encoded = pd.DataFrame(encoded_data, columns=feature_names)
    
    # Rename columns to match
    df_encoded = df_encoded.rename(columns={
        "species": "species_id",
        "island": "island_id"
    })
    
    # Cast IDs to int
    df_encoded["species_id"] = df_encoded["species_id"].astype(int)
    df_encoded["island_id"] = df_encoded["island_id"].astype(int)
    
    return df_encoded

@timeit
def scikit_split_data(df):
    """Splits data using uniform IDs."""
    train_ids = pd.read_parquet(TRAIN_IDS_PARQUET)
    test_ids = pd.read_parquet(TEST_IDS_PARQUET)
    
    train_df = df.merge(train_ids, on="observation_id", how="inner")
    test_df = df.merge(test_ids, on="observation_id", how="inner")
    
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values
    
    return X_train, X_test, y_train, y_test, test_df

@timeit
def scikit_train_model(data):
    """Trains the model on Scikit-prepared data."""
    X_train, X_test, y_train, y_test, _ = data
    model = MODEL.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"  Scikit Accuracy: {accuracy}")
    return model, accuracy

@timeit
def scikit_predict_model(data):
    """Predicts using model.predict()."""
    X_train, X_test, y_train, y_test, test_df = data
    model = MODEL.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
    result_df = test_df[["observation_id", "species_id"]].copy()
    result_df["predicted_species_id"] = predictions
    return result_df

def run_scikit_benchmark():
    df = scikit_load_data()
    df_encoded = scikit_encode_data(df)
    split_data = scikit_split_data(df_encoded)
    _, accuracy = scikit_train_model(split_data)
    predictions = scikit_predict_model(split_data)
    return {"accuracy": accuracy, "predictions": predictions}

# =============================================================================
# --- 5. SPARK BENCHMARK ---
# =============================================================================

@timeit
def spark_load_data(spark):
    """Loads and preprocesses data in Spark."""
    df = (
        spark.read.csv(LOCAL_CSV, header=True, inferSchema=True, nullValue='NA')
        .na.drop()
        .withColumn("observation_id", F.monotonically_increasing_id() + 1)
    )
    
    # Cast types
    df = df.withColumns({
        "bill_length_mm": F.col("bill_length_mm").cast("float"),
        "bill_depth_mm": F.col("bill_depth_mm").cast("float"),
        "body_mass_g": F.col("body_mass_g").cast("int"),
        "flipper_length_mm": F.col("flipper_length_mm").cast("int"),
    })
    return df

@timeit
def spark_encode_data(df):
    """Encodes data using Spark's StringIndexer."""
    # StringIndexer must be alphabetical to match
    indexer_species = StringIndexer(
        inputCol="species", outputCol="species_id", stringOrderType="alphabetAsc"
    ).fit(df)
    df = indexer_species.transform(df)
    
    indexer_island = StringIndexer(
        inputCol="island", outputCol="island_id", stringOrderType="alphabetAsc"
    ).fit(df)
    df = indexer_island.transform(df)
    
    # Cast IDs to int for sklearn compatibility
    df = df.withColumns({
        "species_id": F.col("species_id").cast("int"),
        "island_id": F.col("island_id").cast("int"),
    })
    return df

@timeit
def spark_split_data(df):
    """Splits data using uniform IDs."""
    spark = df.sparkSession
    train_ids = spark.read.parquet(TRAIN_IDS_PARQUET)
    test_ids = spark.read.parquet(TEST_IDS_PARQUET)
    
    train_df = df.join(train_ids, on="observation_id", how="inner").cache()
    test_df = df.join(test_ids, on="observation_id", how="inner").cache()
    
    # Collect data to Python for sklearn
    X_train = np.array(train_df.select(FEATURES).collect())
    y_train = np.array(train_df.select(TARGET).collect()).ravel()
    X_test = np.array(test_df.select(FEATURES).collect())
    y_test = np.array(test_df.select(TARGET).collect()).ravel()
    
    return X_train, X_test, y_train, y_test, test_df

@timeit
def spark_train_model(data):
    """Trains the model on Spark-prepared data."""
    X_train, X_test, y_train, y_test, _ = data
    model = MODEL.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"  Spark Accuracy: {accuracy}")
    return model, accuracy

@timeit
def spark_predict_model(data):
    """Predicts using a Spark Pandas UDF."""
    X_train, X_test, y_train, y_test, test_df = data
    model = MODEL.fit(X_train, y_train)
    
    # Broadcast the model to all executors
    broadcasted_model = test_df.sparkSession.sparkContext.broadcast(model)

    @F.pandas_udf("int")
    def predict_udf(
        bill_length_mm: pd.Series,
        bill_depth_mm: pd.Series,
        flipper_length_mm: pd.Series,
        body_mass_g: pd.Series,
        island_id: pd.Series,
    ) -> pd.Series:
        model = broadcasted_model.value
        X = pd.concat(
            [
                bill_length_mm,
                bill_depth_mm,
                flipper_length_mm,
                body_mass_g,
                island_id,
            ],
            axis=1,
        ).values
        return pd.Series(model.predict(X))

    predictions_df = test_df.withColumn(
        "predicted_species_id",
        predict_udf(
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "island_id",
        ),
    )
    
    return predictions_df.select(
        "observation_id", "species_id", "predicted_species_id"
    ).toPandas()

def run_spark_benchmark(spark: SparkSession):
    df = spark_load_data(spark)
    df_encoded = spark_encode_data(df)
    split_data = spark_split_data(df_encoded)
    _, accuracy = spark_train_model(split_data)
    predictions = spark_predict_model(split_data)
    return {"accuracy": accuracy, "predictions": predictions}