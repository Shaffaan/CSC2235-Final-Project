#!/bin/bash

# --- Fair Benchmarking Harness

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
export PYSPARK_PYTHON="$VENV_PYTHON"
export PYSPARK_DRIVER_PYTHON="$VENV_PYTHON"

# --- Cluster Discovery ---
HOSTNAME=$(hostname)
IS_DISTRIBUTED=false
SPARK_CLUSTER_STARTED=false

if grep -q "node-1" /etc/hosts; then
    IS_DISTRIBUTED=true
fi

if [ "$IS_DISTRIBUTED" = true ] && [[ "$HOSTNAME" != *"node-0"* ]]; then
    echo "Error: Run benchmarks from node-0."
    exit 1
fi

start_spark_cluster() {
    echo "--- [Lazy Start] Setting up Distributed Spark Cluster ---"
    USER_NAME=$(whoami)
    if [ ! -f "$HOME/.ssh/id_rsa" ]; then
        ssh-keygen -t rsa -N "" -f "$HOME/.ssh/id_rsa"
    fi
    for i in {1..4}; do
        cat "$HOME/.ssh/id_rsa.pub" | sudo ssh -o StrictHostKeyChecking=no "node-$i" \
            "mkdir -p /users/$USER_NAME/.ssh && cat >> /users/$USER_NAME/.ssh/authorized_keys"
    done
    $SPARK_HOME/sbin/stop-master.sh
    SPARK_MASTER_HOST=0.0.0.0 $SPARK_HOME/sbin/start-master.sh
    sleep 3
    export SPARK_MASTER_URL="spark://node-0:7077"
    for i in {1..4}; do
        ssh -o StrictHostKeyChecking=no "node-$i" "$SPARK_HOME/sbin/stop-worker.sh; nohup $SPARK_HOME/sbin/start-worker.sh $SPARK_MASTER_URL > /dev/null 2>&1 &"
    done
    sleep 10
    SPARK_CLUSTER_STARTED=true
}

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
        while IFS= read -r script; do
            TARGET_SCRIPTS+=("$script")
        done < <(find "$PIPELINE_DIR" -maxdepth 1 -name "*_pipeline.py" ! -name "__init__.py" | sort)
    else
        TARGET_SCRIPTS=("$PIPELINE_DIR/${FRAMEWORK_NAME_ARG}_pipeline.py")
    fi

    for script in "${TARGET_SCRIPTS[@]}"; do
        if [ ! -f "$script" ]; then continue; fi
        
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
            if [ "$mode" == "distributed" ] && [ "$SPARK_CLUSTER_STARTED" = false ]; then
                start_spark_cluster
            fi

            if [ "$BASE_FRAMEWORK" == "spark" ]; then
                FRAMEWORK_LABEL="spark_${mode}"
            else
                FRAMEWORK_LABEL="$BASE_FRAMEWORK"
            fi
            
            FRAMEWORK_RESULTS_DIR="$RESULTS_PIPELINE_DIR/$FRAMEWORK_LABEL"
            mkdir -p "$FRAMEWORK_RESULTS_DIR"

            echo "Running: $script (Mode: $mode)"
            
            if [ "$SPARK_CLUSTER_STARTED" = true ]; then
                sudo sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
                for i in {1..4}; do
                    ssh -o StrictHostKeyChecking=no node-$i "sudo sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
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
if [ "$SPARK_CLUSTER_STARTED" = true ]; then
    echo "--- Tearing down Spark Cluster ---"
    $SPARK_HOME/sbin/stop-master.sh
    for i in {1..4}; do
        ssh -o StrictHostKeyChecking=no node-$i "$SPARK_HOME/sbin/stop-worker.sh"
    done
fi

# --- Report ---
echo "--- Generating Final Report ---"
"$VENV_PYTHON" common/summarize_results.py "$RESULTS_DIR"