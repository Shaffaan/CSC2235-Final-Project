#!/bin/bash

# --- Fair Benchmarking Harness (Distributed Capable) ---

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

# --- Cluster Discovery ---
HOSTNAME=$(hostname)
IS_DISTRIBUTED=false
MASTER_URL=""

if grep -q "node-1" /etc/hosts; then
    IS_DISTRIBUTED=true
fi

if [ "$IS_DISTRIBUTED" = true ]; then
    if [[ "$HOSTNAME" != *"node-0"* ]]; then
        echo "Error: In a distributed setup, please run benchmarks from node-0."
        exit 1
    fi
    echo "--- Detected Distributed Cluster (5 Nodes) ---"
    
    echo "--- Setting up Passwordless SSH for Cluster ---"
    USER_NAME=$(whoami)
    
    if [ ! -f "$HOME/.ssh/id_rsa" ]; then
        echo "Generating SSH key for $USER_NAME..."
        ssh-keygen -t rsa -N "" -f "$HOME/.ssh/id_rsa"
    fi

    for i in {1..4}; do
        WORKER_NODE="node-$i"
        echo "Propagating public key to $WORKER_NODE..."
        
        cat "$HOME/.ssh/id_rsa.pub" | sudo ssh -o StrictHostKeyChecking=no "$WORKER_NODE" \
            "mkdir -p /users/$USER_NAME/.ssh && cat >> /users/$USER_NAME/.ssh/authorized_keys"
    done
    
    # 1. Start Spark Master on Node 0
    echo "Starting Spark Master on node-0..."
    $SPARK_HOME/sbin/stop-master.sh
    SPARK_MASTER_HOST=0.0.0.0 $SPARK_HOME/sbin/start-master.sh
    
    sleep 3
    MASTER_URL="spark://node-0:7077"
    export SPARK_MASTER_URL="$MASTER_URL"
    
    for i in {1..4}; do
        WORKER_HOST="node-$i"
        echo "Starting Spark Worker on $WORKER_HOST..."
        ssh $WORKER_HOST "$SPARK_HOME/sbin/stop-worker.sh; nohup $SPARK_HOME/sbin/start-worker.sh $MASTER_URL > /dev/null 2>&1 &"
    done
    
    echo "Waiting for workers to register..."
    sleep 10
    
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
        BASE_FRAMEWORK=$(basename "$script" _pipeline.py)
        
        MODES=("standard")
        
        if [ "$BASE_FRAMEWORK" == "spark" ]; then
            if [ "$IS_DISTRIBUTED" = true ]; then
                MODES=("local" "distributed")
            else
                MODES=("local")
            fi
        fi

        for mode in "${MODES[@]}"; do
            
            if [ "$BASE_FRAMEWORK" == "spark" ]; then
                FRAMEWORK_LABEL="spark_${mode}"
            else
                FRAMEWORK_LABEL="$BASE_FRAMEWORK"
            fi
            
            FRAMEWORK_RESULTS_DIR="$RESULTS_PIPELINE_DIR/$FRAMEWORK_LABEL"
            mkdir -p "$FRAMEWORK_RESULTS_DIR"

            echo "Running: $script (Mode: $mode)"
            
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
            export SPARK_MODE="$mode" 

            "$VENV_PYTHON" "$script"
            
            unset SCRIPT_RESULTS_DIR
            unset SPARK_MODE
        done
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