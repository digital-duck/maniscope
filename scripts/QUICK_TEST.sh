#!/bin/bash
# Quick test script for Maniscope migration
# Run from repository root: ./scripts/QUICK_TEST.sh

cd "$(dirname "$0")/.." || exit 1

echo "======================================"
echo "MANISCOPE QUICK TEST"
echo "======================================"
echo ""

# Test 1: Package import
echo "Test 1: Package Import..."
python -c "from maniscope import ManiscopeEngine_v2o; print('✅ PASS')" 2>&1

# Test 2: Engine functionality
echo "Test 2: Engine Functionality..."
python -c "
from maniscope import ManiscopeEngine_v2o
engine = ManiscopeEngine_v2o(k=3, alpha=0.3, verbose=False)
engine.fit(['test1', 'test2', 'test3'])
results = engine.search('test', top_n=1)
print('✅ PASS' if len(results) == 1 else '❌ FAIL')
" 2>&1

# Test 3: Structure
echo "Test 3: Directory Structure..."
if [ -f "ui/Maniscope.py" ] && [ -d "data" ]; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

# Test 4: Datasets
echo "Test 4: Datasets..."
dataset_count=$(ls data/dataset-*.json 2>/dev/null | wc -l)
if [ "$dataset_count" -eq 12 ]; then
    echo "✅ PASS (found $dataset_count datasets)"
else
    echo "❌ FAIL (found $dataset_count datasets, expected 12)"
fi

# Test 5: No old paths
echo "Test 5: Path References..."
if grep -r "parent.parent.parent.parent" ui/ 2>/dev/null | grep -q .; then
    echo "❌ FAIL (found old 4-level parent paths)"
else
    echo "✅ PASS"
fi

echo ""
echo "======================================"
echo "Quick test complete!"
echo "For full testing, see MIGRATION_REVIEW.md"
echo "To launch app: python run_app.py"
echo "======================================"
