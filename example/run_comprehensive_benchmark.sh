#!/bin/bash
#
# Comprehensive Synthesizer Benchmark Script
#
# This script runs all synthesizers on all 4 preprocessed datasets with realistic
# hyperparameters. Includes fixes for data quality issues and error handling.
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
    "preprocessed_conversions_all_8-1-25.csv"
    "preprocessed_sponsored_ads_traffic_7-29-25.csv" 
    "preprocessed_dsp_impressions_7-29-25.csv"
    "preprocessed_amazon_attributed_events_by_traffic_time_7-29-25.csv"
)

# Check if we're in the right directory
if [[ ! -f "run_comprehensive_benchmark.py" ]]; then
    echo "❌ Error: run_comprehensive_benchmark.py not found"
    echo "💡 Please run this script from the /example directory"
    exit 1
fi

if [[ ! -d "sandbox_preprocessed" ]]; then
    echo "❌ Error: sandbox_preprocessed/ directory not found"
    echo "💡 Please ensure sandbox_preprocessed/ directory with CSV files exists"
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
echo "📊 Using preprocessed datasets in sandbox_preprocessed/"
echo "🧬 Testing ALL synthesizers with realistic hyperparameters for FULL data"
echo "📋 Features:"
echo "   - Train/test split: 80/20 with seed=42"
echo "   - Generate synthetic data same size as training data"
echo "   - Save CSV files for all successful synthesizers"
echo "   - Realistic epochs: AutoDiff(50), TabSyn(50), GReaT(30), CTGAN(100), TVAE(100)"
echo "⏱️ No time limits - comprehensive evaluation"
echo "🕐 Start time: $(date)"
echo ""

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Log file for the entire run
LOG_FILE="${BASE_OUTPUT_DIR}/comprehensive_benchmark.log"
echo "📝 Log file: ${LOG_FILE}"
echo ""

# Run the comprehensive Python benchmark
TOTAL_START_TIME=$(date +%s)

if python run_comprehensive_benchmark.py "${BASE_OUTPUT_DIR}" 2>&1 | tee "${LOG_FILE}"; then
    TOTAL_END_TIME=$(date +%s)
    TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
    
    echo ""
    echo "🎉 COMPREHENSIVE BENCHMARK COMPLETE"
    echo "===================================="
    echo "🕐 Total time: $((TOTAL_DURATION/3600))h $((TOTAL_DURATION%3600/60))m"
    echo "📁 All results saved in: ${BASE_OUTPUT_DIR}/"
    echo "   - comprehensive_benchmark_results.json (intermediate)"
    echo "   - comprehensive_benchmark_final.json (final summary)"
    echo "   - synthetic_*.csv files for each successful synthesizer"
    echo "📝 Complete log: ${LOG_FILE}"
    
else
    echo "❌ Comprehensive benchmark failed"
    exit 1
fi

echo ""
echo "🎯 Use these commands to analyze results:"
echo "   cat ${BASE_OUTPUT_DIR}/comprehensive_benchmark_final.json | python -m json.tool"
echo "   ls ${BASE_OUTPUT_DIR}/"

exit 0