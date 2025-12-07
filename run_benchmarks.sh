#!/bin/bash

# --- Fair Benchmarking Harness (Distributed Capable) ---
#
# USAGE:
#   ./run_benchmarks.sh                         (Runs all pipelines)
#   ./run_benchmarks.sh <pipeline> <framework>  (Runs specific config)

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_HOME=/opt/spark

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

PIPELINE_NAME_ARG="$1"
FRAMEWORK_NAME_ARG="$2"

TIMESTAMP=$(date -u +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

VENV_PYTHON="/local/repository/.venv/bin/python3"
export PYTHONPATH=$SCRIPT_DIR

# --- Cluster Discovery ---
# We assume hostnames node-0, node-1, etc. based on Profile.
# 'node-0' is the master.
HOSTNAME=$(hostname)
IS_DISTRIBUTED=false
MASTER_URL=""

# Check if node-1 exists in /etc/hosts to determine if we are in a cluster
if grep -q "node-1" /etc/hosts; then
    IS_DISTRIBUTED=true
fi

if [ "$IS_DISTRIBUTED" = true ]; then
    if [[ "$HOSTNAME" != *"node-0"* ]]; then
        echo "Error: In a distributed setup, please run benchmarks from node-0."
        exit 1
    fi
    echo "--- Detected Distributed Cluster (5 Nodes) ---"
    
    # 1. Start Spark Master on Node 0
    echo "Starting Spark Master on node-0..."
    $SPARK_HOME/sbin/stop-master.sh
    $SPARK_HOME/sbin/start-master.sh
    
    # Wait a moment for master to bind
    sleep 3
    MASTER_URL="spark://node-0:7077"
    export SPARK_MASTER_URL="$MASTER_URL"
    
    # 2. Start Spark Workers on Node 1, 2, 3, 4
    # We assume passwordless SSH is set up by CloudLab/Emulab automatically.
    for i in {1..4}; do
        WORKER_HOST="node-$i"
        echo "Starting Spark Worker on $WORKER_HOST..."
        # Stop existing worker just in case, then start new one pointing to master
        ssh $WORKER_HOST "$SPARK_HOME/sbin/stop-worker.sh; nohup $SPARK_HOME/sbin/start-worker.sh $MASTER_URL > /dev/null 2>&1 &"
    done
    
    # Give workers time to register
    echo "Waiting for workers to register..."
    sleep 10
    
    # Ensure PySpark uses the venv on workers
    export PYSPARK_PYTHON="$VENV_PYTHON"
    export PYSPARK_DRIVER_PYTHON="$VENV_PYTHON"

else
    echo "--- Detected Single Node Setup ---"
fi

# --- Pipeline Identification ---
TARGET_PIPELINES=()
if [ -z "$PIPELINE_NAME_ARG" ]; then
    for p_dir in pipelines/*; do
        if [ -d "$p_dir" ]; then
            TARGET_PIPELINES+=("$(basename "$p_dir")")
        fi
    done
else
    TARGET_PIPELINES=("$PIPELINE_NAME_ARG")
fi

# --- Execution Loop ---
for PIPELINE_NAME in "${TARGET_PIPELINES[@]}"; do
    echo "=== STARTING PIPELINE: $PIPELINE_NAME ==="
    PIPELINE_DIR="pipelines/$PIPELINE_NAME"
    RESULTS_PIPELINE_DIR="$RESULTS_DIR/$PIPELINE_NAME"
    mkdir -p "$RESULTS_PIPELINE_DIR"

    TARGET_SCRIPTS=()
    if [ -z "$FRAMEWORK_NAME_ARG" ]; then
        for script in $(find "$PIPELINE_DIR" -maxdepth 1 -name "*_pipeline.py" ! -name "__init__.py"); do
            TARGET_SCRIPTS+=("$script")
        done
    else
        TARGET_SCRIPTS=("$PIPELINE_DIR/${FRAMEWORK_NAME_ARG}_pipeline.py")
    fi

    for script in "${TARGET_SCRIPTS[@]}"; do
        FRAMEWORK=$(basename "$script" _pipeline.py)
        
        FRAMEWORK_RESULTS_DIR="$RESULTS_PIPELINE_DIR/$FRAMEWORK"
        mkdir -p "$FRAMEWORK_RESULTS_DIR"

        echo "Running: $script"
        
        # Clear Cache on ALL nodes if distributed
        if [ "$IS_DISTRIBUTED" = true ]; then
            echo "Clearing cache on cluster..."
            sudo sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
            for i in {1..4}; do
                ssh node-$i "sudo sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
            done
        else
            sudo sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
        fi

        export SCRIPT_RESULTS_DIR="$FRAMEWORK_RESULTS_DIR"
        "$VENV_PYTHON" "$script"
        unset SCRIPT_RESULTS_DIR
    done
done

# --- Teardown ---
if [ "$IS_DISTRIBUTED" = true ]; then
    echo "--- Tearing down Spark Cluster ---"
    $SPARK_HOME/sbin/stop-master.sh
    for i in {1..4}; do
        ssh node-$i "$SPARK_HOME/sbin/stop-worker.sh"
    done
fi

# --- Report ---
echo "--- Generating Final Report ---"
"$VENV_PYTHON" common/summarize_results.py "$RESULTS_DIR"