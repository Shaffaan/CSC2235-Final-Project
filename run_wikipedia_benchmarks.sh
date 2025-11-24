#!/bin/bash

set -euo pipefail

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%d_%H-%M-%S")
RESULTS_BASE="results/${TIMESTAMP}/wikipedia_data"
mkdir -p "$RESULTS_BASE"
echo "Wikipedia benchmark results will be stored in: $RESULTS_BASE"

VENV_PYTHON="/local/repository/venv/bin/python3"

FRAMEWORKS=("pandas" "polars" "duckdb")

for FRAMEWORK in "${FRAMEWORKS[@]}"; do
    SCRIPT_PATH="pipelines/wikipedia_data/${FRAMEWORK}_pipeline.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Skipping missing script: $SCRIPT_PATH"
        continue
    fi

    FRAMEWORK_RESULTS_DIR="${RESULTS_BASE}/${FRAMEWORK}"
    mkdir -p "$FRAMEWORK_RESULTS_DIR"

    echo ""
    echo "========================================================"
    echo "Running Wikipedia ${FRAMEWORK} pipeline"
    echo "  Script : $SCRIPT_PATH"
    echo "  Results: $FRAMEWORK_RESULTS_DIR"
    echo "========================================================"

    export SCRIPT_RESULTS_DIR="$FRAMEWORK_RESULTS_DIR"
    "$VENV_PYTHON" "$SCRIPT_PATH"
    unset SCRIPT_RESULTS_DIR
done

echo ""
echo "Generating Wikipedia join comparison plots..."
"$VENV_PYTHON" common/summarize_results_wikipedia.py "$RESULTS_BASE"

echo ""
echo "All Wikipedia benchmarks completed."
echo "Artifacts are available under $RESULTS_BASE"
