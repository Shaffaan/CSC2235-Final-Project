#!/bin/bash

# --- Fair Benchmarking Harness ---
#
# USAGE:
#   ./run_benchmarks.sh                          (Runs all pipelines and all frameworks)
#   ./run_benchmarks.sh <pipeline_name>          (Runs all frameworks for one pipeline)
#   ./run_benchmarks.sh <pipeline_name> <framework_name> (Runs one framework for one pipeline)
#
# NOTE: Run this script *without* sudo. It will call sudo itself
#       only for the one command that needs it (clearing cache).

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [<pipeline_name> [<framework_name>]]"
    echo "Example (all): $0"
    echo "Example (one pipeline): $0 nyc_taxi"
    echo "Example (one framework): $0 nyc_taxi duckdb"
    exit 1
fi

PIPELINE_NAME_ARG="$1"
FRAMEWORK_NAME_ARG="$2"

TIMESTAMP=$(date -u +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"

VENV_PYTHON="/local/repository/venv/bin/python3"

TARGET_PIPELINES=()
if [ -z "$PIPELINE_NAME_ARG" ]; then
    echo "Running all pipelines..."
    for p_dir in pipelines/*; do
        if [ -d "$p_dir" ]; then
            TARGET_PIPELINES+=("$(basename "$p_dir")")
        fi
    done
else
    PIPELINE_DIR="pipelines/$PIPELINE_NAME_ARG"
    if [ ! -d "$PIPELINE_DIR" ]; then
        echo "Error: Pipeline directory not found: $PIPELINE_DIR"
        exit 1
    fi
    echo "Running single pipeline: $PIPELINE_NAME_ARG"
    TARGET_PIPELINES=("$PIPELINE_NAME_ARG")
fi

echo "Found ${#TARGET_PIPELINES[@]} pipeline(s) to run: ${TARGET_PIPELINES[*]}"


for PIPELINE_NAME in "${TARGET_PIPELINES[@]}"; do
    echo ""
    echo "========================================================"
    echo "=== STARTING PIPELINE: $PIPELINE_NAME ==="
    echo "========================================================"
    
    PIPELINE_DIR="pipelines/$PIPELINE_NAME"
    RESULTS_PIPELINE_DIR="$RESULTS_DIR/$PIPELINE_NAME"
    mkdir -p "$RESULTS_PIPELINE_DIR"

    TARGET_SCRIPTS=()
    if [ -z "$FRAMEWORK_NAME_ARG" ]; then
        echo "Running all frameworks for pipeline: $PIPELINE_NAME"
        for script in $(find "$PIPELINE_DIR" -maxdepth 1 -name "*_pipeline.py" ! -name "__init__.py"); do
            TARGET_SCRIPTS+=("$script")
        done
    else
        if [ "$PIPELINE_NAME" == "$PIPELINE_NAME_ARG" ]; then
            echo "Running framework '$FRAMEWORK_NAME_ARG' for pipeline: $PIPELINE_NAME"
            SCRIPT_PATH="$PIPELINE_DIR/${FRAMEWORK_NAME_ARG}_pipeline.py"
            if [ ! -f "$SCRIPT_PATH" ]; then
                echo "Error: Framework script not found: $SCRIPT_PATH"
                continue
            fi
            TARGET_SCRIPTS=("$SCRIPT_PATH")
        else
            echo "Skipping pipeline '$PIPELINE_NAME' (waiting for '$PIPELINE_NAME_ARG')"
            continue
        fi
    fi

    echo "Found ${#TARGET_SCRIPTS[@]} script(s) to run."

    for script in "${TARGET_SCRIPTS[@]}"; do
        FRAMEWORK=$(basename "$script" _pipeline.py)
        FRAMEWORK_RESULTS_DIR="$RESULTS_PIPELINE_DIR/$FRAMEWORK"
        
        mkdir -p "$FRAMEWORK_RESULTS_DIR"

        echo ""
        echo "--------------------------------------------------------"
        echo "--- Running: $script ---"
        echo "--- Results Dir: $FRAMEWORK_RESULTS_DIR ---"
        echo "--------------------------------------------------------"

        echo "Step 1: Clearing OS Page Cache (requires sudo)..."
        sudo sync
        sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
        if [ $? -ne 0 ]; then
            echo "!!! Failed to clear OS cache. Ensure user $(whoami) has passwordless sudo."
            exit 1
        fi
        echo "  Caches cleared."

        echo "Step 2: Running benchmark as user '$(whoami)'..."
        
        export SCRIPT_RESULTS_DIR="$FRAMEWORK_RESULTS_DIR"
        
        "$VENV_PYTHON" "$script"
        
        if [ $? -ne 0 ]; then
            echo "!!! Benchmark script failed: $script"
        else
            echo "--- Finished: $script ---"
        fi
        
        unset SCRIPT_RESULTS_DIR
    done

    echo "=== FINISHED PIPELINE: $PIPELINE_NAME ==="

done

echo ""
echo "========================================================"
echo "--- Generating Final Report ---"
echo "========================================================"

"$VENV_PYTHON" common/summarize_results.py "$RESULTS_DIR"

echo "--- All complete. Results are in $RESULTS_DIR ---"