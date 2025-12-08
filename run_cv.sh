#!/bin/bash

# --- Computer Vision Benchmark Harness ---

set -euo pipefail

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

PIPELINE_DIR="pipelines/cv"
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Missing pipeline directory $PIPELINE_DIR"
    exit 1
fi

echo "=== STARTING CV PIPELINES ==="

TARGET_SCRIPTS=()
DEFAULT_FRAMEWORKS=("pandas" "polars" "duckdb")

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

    FRAMEWORK_LABEL=$(basename "$script" _pipeline.py)
    FRAMEWORK_RESULTS_DIR="$RESULTS_DIR/$FRAMEWORK_LABEL"
    mkdir -p "$FRAMEWORK_RESULTS_DIR"

    echo "Running: $script"

    sudo sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

    export SCRIPT_RESULTS_DIR="$FRAMEWORK_RESULTS_DIR"
    "$VENV_PYTHON" "$script"
    unset SCRIPT_RESULTS_DIR
done

echo "--- Generating CV summary ---"
"$VENV_PYTHON" common/summarize_results_cv.py "$RESULTS_DIR"

echo "--- CV pipelines completed. Artifacts stored under $RESULTS_DIR ---"
