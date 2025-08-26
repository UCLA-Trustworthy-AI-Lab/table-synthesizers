#!/bin/bash
#
# Comprehensive Synthesizer Benchmark Script
#
# This script runs all synthesizers on all 4 sandbox datasets with production-ready
# hyperparameters. Estimated total runtime: 8-12 hours depending on hardware.
#
# Usage:
#   chmod +x run_comprehensive_benchmark.sh
#   ./run_comprehensive_benchmark.sh
#
# Notes:
# - This script activates the conda environment 'table-synthesizers'.
# - There is no per-synthesizer time limit; progress logs show elapsed time.
#

set -e  # Exit on any error
set -o pipefail  # Fail if any part of a pipeline fails

# Configuration
TIMEOUT=0  # No time limit per synthesizer
BASE_OUTPUT_DIR="comprehensive_results_$(date +%Y%m%d_%H%M%S)"
DATASETS=(
    "conversions_all_8-1-25.csv"
    "sponsored_ads_traffic_7-29-25.csv" 
    "dsp_impressions_7-29-25.csv"
    "amazon_attributed_events_by_traffic_time_7-29-25.csv"
)

# Check if we're in the right directory
if [[ ! -f "benchmark_synthesizers.py" ]]; then
    echo "❌ Error: benchmark_synthesizers.py not found"
    echo "💡 Please run this script from the /example directory"
    exit 1
fi

if [[ ! -d "sandbox" ]]; then
    echo "❌ Error: sandbox/ directory not found"
    echo "💡 Please ensure sandbox/ directory with CSV files exists"
    exit 1
fi

# Activate conda environment 'table-synthesizers' if available
if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" || true
    if ! conda activate table-synthesizers; then
        echo "❌ Error: conda environment 'table-synthesizers' not found"
        exit 1
    fi
    echo "🧪 Activated conda env: table-synthesizers"
else
    echo "⚠️  Warning: conda not found; proceeding without activating 'table-synthesizers'"
fi

echo "🚀 COMPREHENSIVE SYNTHESIZER BENCHMARK"
echo "========================================"
echo "📊 Datasets: ${#DATASETS[@]}"
echo "⏱️ Timeout per synthesizer: unlimited"
echo "📁 Base output directory: ${BASE_OUTPUT_DIR}"
echo "🕐 Start time: $(date)"
echo "📈 Estimated total time: 8-12 hours"
echo ""

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Log file for the entire run
LOG_FILE="${BASE_OUTPUT_DIR}/comprehensive_benchmark.log"
echo "📝 Comprehensive log: ${LOG_FILE}"
echo ""

# Initialize log
echo "Comprehensive Synthesizer Benchmark" > "${LOG_FILE}"
echo "Started: $(date)" >> "${LOG_FILE}"
echo "Timeout per synthesizer: unlimited" >> "${LOG_FILE}"
echo "Datasets: ${DATASETS[*]}" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# Track overall statistics
TOTAL_START_TIME=$(date +%s)
DATASET_COUNT=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    DATASET_COUNT=$((DATASET_COUNT + 1))
    DATASET_START_TIME=$(date +%s)
    
    echo "[${DATASET_COUNT}/${#DATASETS[@]}] 🔄 PROCESSING: ${dataset}"
    echo "================================================="
    
    # Check if dataset exists
    if [[ ! -f "sandbox/${dataset}" ]]; then
        echo "  ❌ Dataset not found: sandbox/${dataset}"
        echo "Dataset ${dataset}: NOT FOUND" >> "${LOG_FILE}"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        continue
    fi
    
    # Create output directory for this dataset
    DATASET_OUTPUT_DIR="${BASE_OUTPUT_DIR}/results_${dataset%.*}"
    mkdir -p "${DATASET_OUTPUT_DIR}"
    
    echo "  📁 Output: ${DATASET_OUTPUT_DIR}"
    echo "  ⏱️ Started: $(date)"
    echo ""
    
    # Determine appropriate sample size based on dataset
    case "${dataset}" in
        "conversions_all_8-1-25.csv")
            SAMPLES=5000  # Small dataset, use all
            ;;
        "sponsored_ads_traffic_7-29-25.csv")
            SAMPLES=5000  # Medium dataset
            ;;
        "dsp_impressions_7-29-25.csv")
            SAMPLES=5000  # Large dataset, sample down
            ;;
        "amazon_attributed_events_by_traffic_time_7-29-25.csv")
            SAMPLES=4000  # High-dimensional, moderate size
            ;;
        *)
            SAMPLES=5000  # Default
            ;;
    esac
    
    # Run benchmark for this dataset
    echo "  🎯 Running benchmark with ${SAMPLES} samples..."
    
    # Log dataset start
    echo "" >> "${LOG_FILE}"
    echo "=== DATASET: ${dataset} ===" >> "${LOG_FILE}"
    echo "Started: $(date)" >> "${LOG_FILE}"
    echo "Samples: ${SAMPLES}" >> "${LOG_FILE}"
    echo "Output: ${DATASET_OUTPUT_DIR}" >> "${LOG_FILE}"
    
    # Start per-synthesizer elapsed-time monitor
    STATUS_FILE="${DATASET_OUTPUT_DIR}/.current_synth_status"
    : > "${STATUS_FILE}"
    (
        while true; do
            if [[ -s "${STATUS_FILE}" ]]; then
                read -r CURRENT_NAME START_TS < "${STATUS_FILE}"
                if [[ -n "${CURRENT_NAME}" && -n "${START_TS}" ]]; then
                    NOW_TS=$(date +%s)
                    ELAPSED=$(( NOW_TS - START_TS ))
                    printf "  ⏳ %s running for %dm %ds\n" "${CURRENT_NAME}" $((ELAPSED/60)) $((ELAPSED%60)) | tee -a "${LOG_FILE}"
                fi
            fi
            sleep 60
        done
    ) &
    MONITOR_PID=$!

    if stdbuf -oL -eL python benchmark_synthesizers.py \
        --dataset "${dataset}" \
        --n_samples ${SAMPLES} \
        --timeout ${TIMEOUT} \
        --output_dir "${DATASET_OUTPUT_DIR}" \
        --all_synthesizers \
        --verbose 2>&1 | tee -a "${LOG_FILE}" | awk -v status="${STATUS_FILE}" '{
            print
            if ($0 ~ /Running /) {
                name = $0
                sub(/^.*Running /, "", name)
                sub(/\.\.\..*$/, "", name)
                printf("%s %d\n", name, systime()) > status
                close(status)
            } else if ($0 ~ / completed in / || $0 ~ / failed after /) {
                print "" > status
                close(status)
            }
        }'; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        
        DATASET_END_TIME=$(date +%s)
        DATASET_DURATION=$((DATASET_END_TIME - DATASET_START_TIME))
        
        echo "  ✅ Completed in $((DATASET_DURATION/60)) minutes"
        echo "Completed: $(date) (${DATASET_DURATION}s)" >> "${LOG_FILE}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Copy the benchmark report to main directory with dataset prefix
        if [[ -f "${DATASET_OUTPUT_DIR}/benchmark_report.json" ]]; then
            cp "${DATASET_OUTPUT_DIR}/benchmark_report.json" "${BASE_OUTPUT_DIR}/${dataset%.*}_report.json"
        fi
        
    else
        kill "${MONITOR_PID}" 2>/dev/null || true
        echo "  ❌ Failed"
        echo "FAILED: $(date)" >> "${LOG_FILE}"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi
    
    echo ""
done

# Final summary
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo "🎉 COMPREHENSIVE BENCHMARK COMPLETE"
echo "===================================="
echo "🕐 Total time: $((TOTAL_DURATION/3600))h $((TOTAL_DURATION%3600/60))m"
echo "📊 Results:"
echo "   ✅ Successful datasets: ${SUCCESS_COUNT}/${DATASET_COUNT}"
echo "   ❌ Failed datasets: ${FAILURE_COUNT}/${DATASET_COUNT}"
echo "📁 All results saved in: ${BASE_OUTPUT_DIR}"
echo "📝 Complete log: ${LOG_FILE}"

# Write final summary to log
echo "" >> "${LOG_FILE}"
echo "=== FINAL SUMMARY ===" >> "${LOG_FILE}"
echo "Completed: $(date)" >> "${LOG_FILE}"
echo "Total duration: ${TOTAL_DURATION}s ($((TOTAL_DURATION/60)) minutes)" >> "${LOG_FILE}"
echo "Successful datasets: ${SUCCESS_COUNT}/${DATASET_COUNT}" >> "${LOG_FILE}"
echo "Failed datasets: ${FAILURE_COUNT}/${DATASET_COUNT}" >> "${LOG_FILE}"

# Create master summary JSON
SUMMARY_FILE="${BASE_OUTPUT_DIR}/master_summary.json"
cat > "${SUMMARY_FILE}" << EOF
{
    "comprehensive_benchmark_summary": {
        "start_time": "$(date -d @${TOTAL_START_TIME} --iso-8601=seconds)",
        "end_time": "$(date -d @${TOTAL_END_TIME} --iso-8601=seconds)",
        "total_duration_seconds": ${TOTAL_DURATION},
        "total_datasets": ${DATASET_COUNT},
        "successful_datasets": ${SUCCESS_COUNT},
        "failed_datasets": ${FAILURE_COUNT},
        "success_rate": $(echo "scale=3; ${SUCCESS_COUNT}/${DATASET_COUNT}" | bc -l),
        "timeout_per_synthesizer": 0,
        "datasets_processed": [$(printf '"%s",' "${DATASETS[@]}" | sed 's/,$//')],
        "output_directory": "${BASE_OUTPUT_DIR}",
        "log_file": "${LOG_FILE}"
    }
}
EOF

echo "📋 Master summary: ${SUMMARY_FILE}"
echo ""

# Show quick results overview
if [[ ${SUCCESS_COUNT} -gt 0 ]]; then
    echo "📈 QUICK RESULTS OVERVIEW:"
    for dataset in "${DATASETS[@]}"; do
        REPORT_FILE="${BASE_OUTPUT_DIR}/${dataset%.*}_report.json"
        if [[ -f "${REPORT_FILE}" ]]; then
            SUCCESS_RATE=$(python3 -c "
import json
with open('${REPORT_FILE}') as f:
    data = json.load(f)
    sr = data['results_summary']['success_rate']
    total = data['results_summary']['total_synthesizers']
    successful = data['results_summary']['successful']
    print(f'{successful}/{total} ({sr:.1%})')
" 2>/dev/null || echo "N/A")
            echo "  📊 ${dataset%.*}: ${SUCCESS_RATE} synthesizers successful"
        fi
    done
fi

echo ""
echo "🎯 Use these commands to analyze results:"
echo "   ls ${BASE_OUTPUT_DIR}/*/benchmark_report.json"
echo "   python3 -c \"import json; print(json.dumps(json.load(open('${BASE_OUTPUT_DIR}/master_summary.json')), indent=2))\""

exit 0