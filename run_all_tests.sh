#!/bin/bash

# Configuration
TEST_DIR="test"
PYTHON_CMD="python"  # Or specific python path if needed
LOG_FILE="test_run.log"

# Initialize counters
total_tests=0
passed_tests=0
failed_tests=0
failed_files=()

# Clear previous log
> "$LOG_FILE"

echo "Starting test execution..."
echo "========================================"

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory '$TEST_DIR' not found."
    exit 1
fi

# Iterate through each test file matching the pattern
for test_file in "$TEST_DIR"/test*.py; do
    # Check if files actually exist (handle case where glob matches nothing)
    if [ ! -e "$test_file" ]; then
        echo "No test files found in $TEST_DIR matching 'test*.py'"
        break
    fi

    ((total_tests++))
    echo "Running $test_file..."
    
    # Run the test file using python
    # Capture both stdout and stderr, tee to console and log file
    echo "----------------------------------------" >> "$LOG_FILE"
    echo "Output for $test_file:" >> "$LOG_FILE"
    
    if $PYTHON_CMD "$test_file" >> "$LOG_FILE" 2>&1; then
        echo "✅ PASS: $test_file"
        ((passed_tests++))
    else
        echo "❌ FAIL: $test_file"
        ((failed_tests++))
        failed_files+=("$test_file")
    fi
    echo "----------------------------------------"
done

# Print Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total Tests: $total_tests"
echo "Passed:      $passed_tests"
echo "Failed:      $failed_tests"

if [ $failed_tests -gt 0 ]; then
    echo ""
    echo "Failed Tests:"
    for file in "${failed_files[@]}"; do
        echo " - $file"
    done
    echo ""
    echo "See $LOG_FILE for detailed output."
    exit 1
else
    echo ""
    echo "All tests passed successfully!"
    exit 0
fi
