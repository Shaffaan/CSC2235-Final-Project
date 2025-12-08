#!/bin/bash

# --- Computer Vision Benchmark Harness (with Spark Support) ---

set -euo pipefail

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_HOME=/opt/spark

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

FRAMEWORK_ARGS=("$@")

TIMESTAMP=$(date -u +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="results/$TIMESTAMP/cv"
mkdir -p "$RESULTS_DIR"

VENV_CANDIDATES=(
    "/local/repository/venv/bin/python3"
    "/local/repository/.venv/bin/python3"
)

for candidate in "${VENV_CANDIDATES[@]}"; do
    if [ -x "$candidate" ]; then
        VENV_PYTHON="$candidate"
        break
    fi
done

if [ -z "${VENV_PYTHON:-}" ]; then
    echo "Error: None of these interpreters exist: ${VENV_CANDIDATES[*]}."
    echo "Please create one of them or adjust run_cv.sh."
    exit 1
fi

export PYSPARK_PYTHON="$VENV_PYTHON"
export PYSPARK_DRIVER_PYTHON="$VENV_PYTHON"

HOSTNAME=$(hostname)
IS_DISTRIBUTED=false
SPARK_CLUSTER_STARTED=false

if grep -q "node-1" /etc/hosts; then
    IS_DISTRIBUTED=true
fi

if [ "$IS_DISTRIBUTED" = true ] && [[ "$HOSTNAME" != *"node-0"* ]]; then
    echo "Error: In a distributed setup, please run CV benchmarks from node-0."
    exit 1
fi

if [ "$IS_DISTRIBUTED" = true ]; then
    echo "--- Detected Distributed Environment (multiple nodes available) ---"
else
    echo "--- Detected Single Node Setup ---"
fi

start_spark_cluster() {
    echo "--- [Lazy Start] Setting up Distributed Spark Cluster ---"

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

    echo "Starting Spark Master on node-0..."
    $SPARK_HOME/sbin/stop-master.sh
    SPARK_MASTER_HOST=0.0.0.0 $SPARK_HOME/sbin/start-master.sh

    sleep 3
    MASTER_URL="spark://node-0:7077"
    export SPARK_MASTER_URL="$MASTER_URL"

    for i in {1..4}; do
        WORKER_HOST="node-$i"
        echo "Starting Spark Worker on $WORKER_HOST..."
        ssh -o StrictHostKeyChecking=no "$WORKER_HOST" "$SPARK_HOME/sbin/stop-worker.sh; nohup $SPARK_HOME/sbin/start-worker.sh $MASTER_URL > /dev/null 2>&1 &"
    done

    echo "Waiting for workers to register..."
    sleep 10

    SPARK_CLUSTER_STARTED=true
}

PIPELINE_DIR="pipelines/cv"
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Missing pipeline directory $PIPELINE_DIR"
    exit 1
fi

echo "=== STARTING CV PIPELINES ==="

TARGET_SCRIPTS=()
DEFAULT_FRAMEWORKS=("pandas" "polars" "duckdb" "spark")

if [ ${#FRAMEWORK_ARGS[@]} -eq 0 ] || { [ ${#FRAMEWORK_ARGS[@]} -eq 1 ] && [ "${FRAMEWORK_ARGS[0]}" = "all" ]; }; then
    for framework in "${DEFAULT_FRAMEWORKS[@]}"; do
        script="$PIPELINE_DIR/${framework}_pipeline.py"
        if [ -f "$script" ]; then
            TARGET_SCRIPTS+=("$script")
        else
            echo "Warning: default CV pipeline missing for framework '$framework' ($script)"
        fi
    done
    while IFS= read -r script; do
        [ -n "$script" ] || continue
        already_added=false
        for registered in "${TARGET_SCRIPTS[@]}"; do
            if [ "$registered" = "$script" ]; then
                already_added=true
                break
            fi
        done
        if [ "$already_added" = false ]; then
            TARGET_SCRIPTS+=("$script")
        fi
    done < <(find "$PIPELINE_DIR" -maxdepth 1 -name "*_pipeline.py" ! -name "__init__.py" | sort)
else
    for arg in "${FRAMEWORK_ARGS[@]}"; do
        framework="${arg}"
        script_candidate=""
        if [[ "$framework" == *.py ]] && [ -f "$framework" ]; then
            script_candidate="$framework"
        elif [ -f "$PIPELINE_DIR/$framework" ]; then
            script_candidate="$PIPELINE_DIR/$framework"
        else
            script_candidate="$PIPELINE_DIR/${framework}_pipeline.py"
        fi

        if [ -f "$script_candidate" ]; then
            TARGET_SCRIPTS+=("$script_candidate")
        else
            echo "Skipping missing script: $script_candidate"
        fi
    done
fi

if [ ${#TARGET_SCRIPTS[@]} -eq 0 ]; then
    echo "No pipeline scripts found under $PIPELINE_DIR"
    exit 0
fi

for script in "${TARGET_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "Skipping missing script: $script"
        continue
    fi

    BASE_FRAMEWORK=$(basename "$script" _pipeline.py)
    MODES=("standard")
    if [ "$BASE_FRAMEWORK" = "spark" ]; then
        if [ "$IS_DISTRIBUTED" = true ]; then
            MODES=("local" "distributed")
        else
            MODES=("local")
        fi
    fi

    for mode in "${MODES[@]}"; do
        if [ "$mode" = "distributed" ] && [ "$SPARK_CLUSTER_STARTED" = false ]; then
            start_spark_cluster
        fi

        if [ "$BASE_FRAMEWORK" = "spark" ]; then
            FRAMEWORK_LABEL="spark_${mode}"
        else
            FRAMEWORK_LABEL="$BASE_FRAMEWORK"
        fi
        FRAMEWORK_RESULTS_DIR="$RESULTS_DIR/$FRAMEWORK_LABEL"
        mkdir -p "$FRAMEWORK_RESULTS_DIR"

        echo "Running: $script (Mode: $mode)"

        if [ "$SPARK_CLUSTER_STARTED" = true ]; then
            echo "Clearing cache on cluster (Master + Workers)..."
            sudo sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
            for i in {1..4}; do
                ssh -o StrictHostKeyChecking=no "node-$i" "sudo sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
            done
        else
            sudo sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
        fi

        export SCRIPT_RESULTS_DIR="$FRAMEWORK_RESULTS_DIR"
        if [ "$BASE_FRAMEWORK" = "spark" ]; then
            export SPARK_MODE="$mode"
        fi
        "$VENV_PYTHON" "$script"
        unset SCRIPT_RESULTS_DIR
        unset SPARK_MODE
    done
done

if [ "$SPARK_CLUSTER_STARTED" = true ]; then
    echo "--- Tearing down Spark Cluster ---"
    $SPARK_HOME/sbin/stop-master.sh
    for i in {1..4}; do
        ssh -o StrictHostKeyChecking=no "node-$i" "$SPARK_HOME/sbin/stop-worker.sh"
    done
fi

echo "--- Generating CV summary ---"
"$VENV_PYTHON" common/summarize_results_cv.py "$RESULTS_DIR"

echo "--- CV pipelines completed. Artifacts stored under $RESULTS_DIR ---"
