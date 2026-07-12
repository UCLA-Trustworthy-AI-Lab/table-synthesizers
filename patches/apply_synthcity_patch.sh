#!/bin/bash
# Apply synthcity pgmpy compatibility patch
# This patch fixes compatibility issues between synthcity 0.2.12 and pgmpy 1.0.0

set -e

VENV_PATH="${1:-.venv}"
TARGET_FILE="$VENV_PATH/lib/python3.12/site-packages/synthcity/plugins/generic/plugin_bayesian_network.py"

echo "========================================="
echo "Synthcity pgmpy Compatibility Patcher"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    echo "Usage: $0 [venv_path]"
    exit 1
fi

# Check if target file exists
if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ Error: Target file not found:"
    echo "   $TARGET_FILE"
    echo ""
    echo "This might mean:"
    echo "  1. synthcity is not installed"
    echo "  2. Python version is not 3.12"
    echo "  3. synthcity is installed in a different location"
    exit 1
fi

echo "✓ Found target file: $TARGET_FILE"
echo ""

# Check if already patched
if grep -q "DiscreteBayesianNetwork as BayesianNetwork" "$TARGET_FILE"; then
    echo "✓ Patch already applied!"
    echo ""
    exit 0
fi

echo "Applying patch..."

# Backup original file
BACKUP_FILE="${TARGET_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$TARGET_FILE" "$BACKUP_FILE"
echo "✓ Created backup: $BACKUP_FILE"

# Apply patch using sed
sed -i 's/from pgmpy.models import BayesianNetwork/from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork/' "$TARGET_FILE"
sed -i 's/estimators\.K2Score/estimators.K2/' "$TARGET_FILE"
sed -i 's/estimators\.BDeuScore/estimators.BDeu/' "$TARGET_FILE"
sed -i 's/estimators\.BicScore/estimators.BIC/' "$TARGET_FILE"
sed -i 's/estimators\.BDsScore/estimators.BDs/' "$TARGET_FILE"

echo "✓ Patch applied successfully!"
echo ""

# Verify patch
if grep -q "DiscreteBayesianNetwork as BayesianNetwork" "$TARGET_FILE" && \
   grep -q "estimators.K2," "$TARGET_FILE"; then
    echo "✓ Verification passed!"
    echo ""
    echo "========================================="
    echo "Patch applied successfully!"
    echo "========================================="
    echo ""
    echo "Changes made:"
    echo "  1. BayesianNetwork → DiscreteBayesianNetwork"
    echo "  2. K2Score → K2"
    echo "  3. BDeuScore → BDeu"
    echo "  4. BicScore → BIC"
    echo "  5. BDsScore → BDs"
    echo ""
    echo "Backup saved to: $BACKUP_FILE"
    exit 0
else
    echo "❌ Verification failed!"
    echo "Restoring from backup..."
    cp "$BACKUP_FILE" "$TARGET_FILE"
    echo "✓ Restored original file"
    exit 1
fi
