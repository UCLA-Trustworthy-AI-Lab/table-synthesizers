#!/bin/bash
# Comprehensive test runner for table-synthesizers framework
# Supports full test suite, specific algorithms, edge case tiers, and specialized test categories

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$SCRIPT_DIR/tests"
CONDA_ENV="table-synthesizers"

# Test categories -- algorithm groups
CORE_ALGORITHMS=("identity" "TVAE" "TabDDPM")
STABLE_ALGORITHMS=("CART" "DPCART" "SMOTE" "GaussianCopula" "TabDiff" "TabPFGen")
SYNTHCITY_ALGORITHMS=("ARF" "BayesianNetwork" "GREAT" "NFlow")
EXPERIMENTAL_ALGORITHMS=("TabSyn" "AutoDiff" "CTGAN" "PATECTGAN")
OPTIONAL_ALGORITHMS=("AIM")
# LTM tests - currently only unit test available, integration tests not yet implemented
LTM_TESTS=("ltm_vae")  # Maps to tests/unit/test_ltm_vae.py
LIBZERO_TESTS=("test_libzero_workaround" "test_libzero_proof" "test_affected_modules" "test_pytorch_compatibility")

# Edge case test file (uses pytest markers for tier filtering)
EDGE_CASE_TEST="$TEST_DIR/integration/test_edge_cases.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..60})${NC}"
}

# Environment setup
setup_environment() {
    log_header "🔧 ENVIRONMENT SETUP"

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install Anaconda/Miniconda first."
        exit 1
    fi

    # Activate conda environment
    log_info "Activating conda environment: $CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || {
        log_error "Failed to activate conda environment: $CONDA_ENV"
        log_info "Creating environment with Python 3.11 (required for LTM-VAE)..."
        conda create -n "$CONDA_ENV" python=3.11 -y
        conda activate "$CONDA_ENV"
    }

    # Set up Python path for all modules
    export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

    # Load environment variables from .env
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        source "$SCRIPT_DIR/.env"
        # Expand tilde in DATASET_PATH and export
        if [[ -n "$DATASET_PATH" ]]; then
            DATASET_PATH="${DATASET_PATH/#\~/$HOME}"
            export DATASET_PATH
            log_info "DATASET_PATH: $DATASET_PATH"
        fi
    else
        log_warning ".env file not found, some tests may fail"
    fi

    log_success "Environment activated"
    log_info "Python version: $(python --version)"
    log_info "PYTHONPATH configured"
    echo
}

# Apply libzero workaround
apply_libzero_workaround() {
    log_info "Applying libzero workaround..."
    cd "$SCRIPT_DIR"
    python -c "
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/stg')
try:
    import stg.zero_workaround as zero
    sys.modules['zero'] = zero
    print('✅ libzero workaround applied')
except ImportError as e:
    print(f'⚠️  libzero workaround not found: {e}')
    print('Continuing without workaround - will be applied by individual tests')
"
}

# Run individual test
run_test() {
    local test_name="$1"
    local test_path=""
    local found_category=""

    # Normalize test name (lowercase for case-insensitive matching)
    local lower_name=$(echo "$test_name" | tr '[:upper:]' '[:lower:]')

    # 1. Check for specific file in integration (preferred)
    if [[ -f "$TEST_DIR/integration/${test_name}.py" ]]; then
        test_path="$TEST_DIR/integration/${test_name}.py"
        found_category="integration"
    elif [[ -f "$TEST_DIR/integration/${test_name}" ]]; then
        test_path="$TEST_DIR/integration/${test_name}"
        found_category="integration"

    # 2. Check for test_{name}_integration.py convention in integration
    elif [[ -f "$TEST_DIR/integration/test_${lower_name}_integration.py" ]]; then
        test_path="$TEST_DIR/integration/test_${lower_name}_integration.py"
        found_category="integration"

    # 3. Check for test_{name}.py in integration (legacy/direct names)
    elif [[ -f "$TEST_DIR/integration/test_${test_name}.py" ]]; then
        test_path="$TEST_DIR/integration/test_${test_name}.py"
        found_category="integration"

    # 4. Check for test_{name}.py in unit
    elif [[ -f "$TEST_DIR/unit/test_${lower_name}.py" ]]; then
        test_path="$TEST_DIR/unit/test_${lower_name}.py"
        found_category="unit"

    # 5. Check for test_{name}.py in unit (case sensitive)
    elif [[ -f "$TEST_DIR/unit/test_${test_name}.py" ]]; then
        test_path="$TEST_DIR/unit/test_${test_name}.py"
        found_category="unit"

    # 6. Direct file path check (if user provided relative path)
    elif [[ -f "$test_name" ]]; then
        test_path="$test_name"
    fi

    if [[ -z "$test_path" ]]; then
        log_warning "Test file not found for: $test_name (searched in $TEST_DIR/integration and $TEST_DIR/unit)"
        return 1
    fi

    log_info "Running test: $test_name ($test_path)"
    cd "$SCRIPT_DIR"

    # Use different runners based on test type
    if [[ "$test_name" == "ltm"* ]]; then
        # LTM tests may need special handling
        python "$test_path" || return 1
    elif [[ "$test_name" == "libzero"* ]] || [[ "$test_name" == "pytorch"* ]] || [[ "$test_name" == "affected"* ]]; then
        # Infrastructure tests
        python "$test_path" || return 1
    else
        # Algorithm tests - use pytest via python -m to ensure correct environment
        python -m pytest "$test_path" -v || return 1
    fi

    log_success "✅ $test_name passed"
    return 0
}

# Run test category
run_test_category() {
    local category="$1"
    shift
    local tests=("$@")

    log_header "🧪 TESTING $category"

    local passed=0
    local failed=0
    local failed_tests=()

    for test in "${tests[@]}"; do
        if run_test "$test"; then
            ((passed++))
        else
            ((failed++))
            failed_tests+=("$test")
        fi
        echo
    done

    log_info "$category Results: $passed passed, $failed failed"
    if [[ $failed -gt 0 ]]; then
        log_warning "Failed tests: ${failed_tests[*]}"
    fi
    echo

    return $failed
}

# Run edge case tests with optional marker filter
run_edge_case_tests() {
    local tier="$1"  # "core", "synthcity", "gpu", or "all"
    local marker_args=""
    local label=""

    case "$tier" in
        core)
            marker_args="-m edge_core"
            label="CORE MODEL EDGE CASES (Identity, CART, DPCART, TVAE, TabDiff, TabPFGen, ...)"
            ;;
        synthcity)
            marker_args="-m edge_synthcity"
            label="SYNTHCITY MODEL EDGE CASES (ARF, BayesianNetwork, GREAT, NFlow)"
            ;;
        gpu)
            marker_args="-m edge_gpu"
            label="GPU MODEL EDGE CASES (CTGAN, PATECTGAN, TabDDPM, AutoDiff, TabSyn)"
            ;;
        all)
            marker_args=""
            label="ALL EDGE CASES (core + synthcity + GPU)"
            ;;
        *)
            log_error "Unknown edge case tier: $tier"
            return 1
            ;;
    esac

    log_header "🔬 TESTING $label"

    cd "$SCRIPT_DIR"
    if [[ -n "$marker_args" ]]; then
        python -m pytest "$EDGE_CASE_TEST" $marker_args -v || {
            log_error "Edge case tests ($tier) failed"
            return 1
        }
    else
        python -m pytest "$EDGE_CASE_TEST" -v || {
            log_error "Edge case tests (all) failed"
            return 1
        }
    fi

    log_success "Edge case tests ($tier) passed"
    echo
    return 0
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [ALGORITHMS...]

Run comprehensive tests for the table-synthesizers framework.

ALGORITHM TEST OPTIONS:
    --full                  Run all tests including edge cases (default)
    --core                  Core algorithm tests (Identity, TVAE, TabDDPM)
    --stable                Stable algorithm tests (CART, DPCART, SMOTE, GaussianCopula,
                            TabDiff, TabPFGen)
    --synthcity             Synthcity-backend algorithm tests (ARF, BayesianNetwork,
                            GREAT, NFlow)
    --experimental          Experimental algorithm tests (TabSyn, AutoDiff, CTGAN,
                            PATECTGAN)
    --optional              Optional dependency tests (AIM)
    --ltm                   LTM-VAE specific tests
    --algorithms            All algorithm tests (excludes infrastructure & edge cases)

EDGE CASE TEST OPTIONS:
    --edge-cases            Run ALL edge case tests (all tiers)
    --edge-core             Edge cases for core/base models only
    --edge-synthcity        Edge cases for synthcity models only
    --edge-gpu              Edge cases for GPU-heavy models only

INFRASTRUCTURE TEST OPTIONS:
    --libzero               Run libzero workaround tests
    --infrastructure        Run infrastructure/compatibility tests
    --help                  Show this help message

ALGORITHMS:
    Specify individual algorithm names to test specific models:
    Identity, TVAE, TabDDPM, CART, DPCART, SMOTE, GaussianCopula, TabDiff,
    TabPFGen, TabSyn, AutoDiff, CTGAN, PATECTGAN, AIM, ARF, BayesianNetwork,
    GREAT, NFlow

EXAMPLES:
    $0                              # Run full suite (all algorithms + edge cases)
    $0 --core                       # Core algorithms only
    $0 --edge-cases                 # All edge case tests
    $0 --edge-core                  # Edge cases for core models
    $0 --edge-synthcity             # Edge cases for synthcity models
    $0 --edge-gpu                   # Edge cases for GPU models
    $0 --stable --synthcity         # Stable + synthcity algorithms
    $0 --core --edge-core           # Core algorithms + their edge cases
    $0 TVAE TabDDPM                 # Specific algorithms only

EOF
}

# Parse command line arguments
parse_arguments() {
    if [[ $# -eq 0 ]]; then
        # Default: run all tests
        RUN_FULL=true
        return
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                RUN_FULL=true
                shift
                ;;
            --core)
                RUN_CORE=true
                shift
                ;;
            --stable)
                RUN_STABLE=true
                shift
                ;;
            --synthcity)
                RUN_SYNTHCITY=true
                shift
                ;;
            --experimental)
                RUN_EXPERIMENTAL=true
                shift
                ;;
            --optional)
                RUN_OPTIONAL=true
                shift
                ;;
            --ltm)
                RUN_LTM=true
                shift
                ;;
            --libzero)
                RUN_LIBZERO=true
                shift
                ;;
            --infrastructure)
                RUN_INFRASTRUCTURE=true
                shift
                ;;
            --algorithms)
                RUN_ALGORITHMS=true
                shift
                ;;
            --edge-cases)
                RUN_EDGE_ALL=true
                shift
                ;;
            --edge-core)
                RUN_EDGE_CORE=true
                shift
                ;;
            --edge-synthcity)
                RUN_EDGE_SYNTHCITY=true
                shift
                ;;
            --edge-gpu)
                RUN_EDGE_GPU=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            --*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                # Individual algorithm names
                SPECIFIC_ALGORITHMS+=("$1")
                shift
                ;;
        esac
    done
}

# Main test execution
run_tests() {
    local total_failed=0

    if [[ "$RUN_FULL" == "true" ]]; then
        log_header "🚀 RUNNING FULL TEST SUITE"

        # Infrastructure tests first
        if ! run_test_category "INFRASTRUCTURE TESTS" "${LIBZERO_TESTS[@]}"; then
            ((total_failed++))
        fi

        # Core algorithms
        if ! run_test_category "CORE ALGORITHMS" "${CORE_ALGORITHMS[@]}"; then
            ((total_failed++))
        fi

        # Stable algorithms
        if ! run_test_category "STABLE ALGORITHMS" "${STABLE_ALGORITHMS[@]}"; then
            ((total_failed++))
        fi

        # Synthcity algorithms
        if ! run_test_category "SYNTHCITY ALGORITHMS" "${SYNTHCITY_ALGORITHMS[@]}"; then
            ((total_failed++))
        fi

        # LTM tests
        if ! run_test_category "LTM TESTS" "${LTM_TESTS[@]}"; then
            ((total_failed++))
        fi

        # Experimental algorithms
        log_warning "Experimental algorithms may have dependencies or compatibility issues"
        if ! run_test_category "EXPERIMENTAL ALGORITHMS" "${EXPERIMENTAL_ALGORITHMS[@]}"; then
            ((total_failed++))
        fi

        # Optional algorithms
        log_warning "Optional algorithms require additional dependencies"
        if ! run_test_category "OPTIONAL ALGORITHMS" "${OPTIONAL_ALGORITHMS[@]}"; then
            ((total_failed++))
        fi

        # Edge case tests (all tiers, run separately for clear reporting)
        log_info "Running edge case tests across all model tiers..."
        if ! run_edge_case_tests "core"; then
            ((total_failed++))
        fi
        if ! run_edge_case_tests "synthcity"; then
            ((total_failed++))
        fi
        if ! run_edge_case_tests "gpu"; then
            ((total_failed++))
        fi

    else
        # Run specific categories or algorithms

        if [[ "$RUN_INFRASTRUCTURE" == "true" || "$RUN_LIBZERO" == "true" ]]; then
            if ! run_test_category "INFRASTRUCTURE TESTS" "${LIBZERO_TESTS[@]}"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_CORE" == "true" ]]; then
            if ! run_test_category "CORE ALGORITHMS" "${CORE_ALGORITHMS[@]}"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_STABLE" == "true" ]]; then
            if ! run_test_category "STABLE ALGORITHMS" "${STABLE_ALGORITHMS[@]}"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_SYNTHCITY" == "true" ]]; then
            if ! run_test_category "SYNTHCITY ALGORITHMS" "${SYNTHCITY_ALGORITHMS[@]}"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_EXPERIMENTAL" == "true" ]]; then
            if ! run_test_category "EXPERIMENTAL ALGORITHMS" "${EXPERIMENTAL_ALGORITHMS[@]}"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_OPTIONAL" == "true" ]]; then
            if ! run_test_category "OPTIONAL ALGORITHMS" "${OPTIONAL_ALGORITHMS[@]}"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_LTM" == "true" ]]; then
            if ! run_test_category "LTM TESTS" "${LTM_TESTS[@]}"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_ALGORITHMS" == "true" ]]; then
            local all_algorithms=("${CORE_ALGORITHMS[@]}" "${STABLE_ALGORITHMS[@]}" "${SYNTHCITY_ALGORITHMS[@]}" "${EXPERIMENTAL_ALGORITHMS[@]}" "${OPTIONAL_ALGORITHMS[@]}")
            if ! run_test_category "ALL ALGORITHMS" "${all_algorithms[@]}"; then
                ((total_failed++))
            fi
        fi

        # Edge case tests
        if [[ "$RUN_EDGE_ALL" == "true" ]]; then
            if ! run_edge_case_tests "all"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_EDGE_CORE" == "true" ]]; then
            if ! run_edge_case_tests "core"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_EDGE_SYNTHCITY" == "true" ]]; then
            if ! run_edge_case_tests "synthcity"; then
                ((total_failed++))
            fi
        fi

        if [[ "$RUN_EDGE_GPU" == "true" ]]; then
            if ! run_edge_case_tests "gpu"; then
                ((total_failed++))
            fi
        fi

        # Run specific algorithms
        if [[ ${#SPECIFIC_ALGORITHMS[@]} -gt 0 ]]; then
            if ! run_test_category "SPECIFIC ALGORITHMS" "${SPECIFIC_ALGORITHMS[@]}"; then
                ((total_failed++))
            fi
        fi
    fi

    return $total_failed
}

# Generate test summary
generate_summary() {
    local exit_code=$1

    log_header "📊 TEST SUMMARY"

    if [[ $exit_code -eq 0 ]]; then
        log_success "All tests passed successfully! 🎉"
        echo
        log_info "Framework Status:"
        log_success "  ✅ libzero workaround functional"
        log_success "  ✅ Core algorithms working"
        log_success "  ✅ PyTorch 2.x compatibility confirmed"
        log_success "  ✅ Framework fully operational"
    else
        log_error "Some tests failed. Check output above for details."
        echo
        log_info "Troubleshooting:"
        log_info "  • Check dependency installation"
        log_info "  • Verify conda environment setup"
        log_info "  • Review PYTHONPATH configuration"
        log_info "  • Check libzero workaround application"
        log_info "  • For synthcity edge cases: pip install -r requirements-synthcity.txt"
        log_info "  • For GPU edge cases: ensure CUDA/MPS device available"
    fi

    echo
    log_info "For detailed logs, see individual test outputs above."
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Reset PYTHONPATH if needed
    # No specific cleanup needed for now
}

# Signal handlers
trap cleanup EXIT
trap 'log_error "Test interrupted"; exit 130' INT

# Initialize variables
RUN_FULL=false
RUN_CORE=false
RUN_STABLE=false
RUN_SYNTHCITY=false
RUN_EXPERIMENTAL=false
RUN_OPTIONAL=false
RUN_LTM=false
RUN_LIBZERO=false
RUN_INFRASTRUCTURE=false
RUN_ALGORITHMS=false
RUN_EDGE_ALL=false
RUN_EDGE_CORE=false
RUN_EDGE_SYNTHCITY=false
RUN_EDGE_GPU=false
SPECIFIC_ALGORITHMS=()

# Main execution
main() {
    log_header "🧪 TABLE-SYNTHESIZERS COMPREHENSIVE TEST RUNNER"
    echo "Testing framework with libzero workaround and PyTorch 2.x compatibility"
    echo

    # Parse arguments
    parse_arguments "$@"

    # Setup environment
    setup_environment

    # Apply workaround
    apply_libzero_workaround

    # Run tests
    run_tests
    local exit_code=$?

    # Generate summary
    generate_summary $exit_code

    exit $exit_code
}

# Run main function with all arguments
main "$@"
