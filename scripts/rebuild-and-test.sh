#!/bin/bash
# FastTracker: Complete Rebuild and Test Workflow
# This script automates the mandatory workflow for code changes

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================="
echo "FastTracker: Rebuild and Test Workflow"
echo "========================================="
echo ""

# Step 1: Docker Force Rebuild
echo "=== Step 1/5: Docker Force Rebuild (--no-cache) ==="
docker compose build --no-cache 2>&1 | tee /tmp/fasttracker_build.log | tail -30

# Verify build success
if grep -q "Built target fasttracker" /tmp/fasttracker_build.log; then
    echo "✅ Build successful"
else
    echo "❌ Build failed - check /tmp/fasttracker_build.log"
    exit 1
fi
echo ""

# Step 2: Container Full Restart
echo "=== Step 2/5: Container Full Restart ==="
docker compose down
sleep 1
docker compose up -d
sleep 2
echo "✅ Container restarted"
echo ""

# Step 3: Binary Verification
echo "=== Step 3/5: Binary Verification ==="
BINARY_INFO=$(docker exec fasttracker-fasttracker-1 stat -c '%y %n' /app/fasttracker)
echo "Binary: $BINARY_INFO"
BINARY_TIME=$(docker exec fasttracker-fasttracker-1 stat -c '%Y' /app/fasttracker)
CURRENT_TIME=$(date +%s)
TIME_DIFF=$((CURRENT_TIME - BINARY_TIME))

if [ $TIME_DIFF -lt 300 ]; then
    echo "✅ Binary is recent (${TIME_DIFF}s old)"
else
    echo "⚠️  Warning: Binary is ${TIME_DIFF}s old (may not be updated)"
fi
echo ""

# Step 4: Run Test Simulation
echo "=== Step 4/5: Test Simulation ==="
docker exec fasttracker-fasttracker-1 /app/fasttracker \
  --mode tracker \
  --scenario single-ballistic \
  --duration 10 \
  --framerate 1 \
  --launch-x 0 --launch-y 0 \
  --target-x 20000 --target-y 20000 \
  --sensor-x 10000 --sensor-y 10000 \
  --radar-max-range 50000 \
  --radar-fov 3.14159 \
  --antenna-boresight -2.356 \
  --search-center -2.356 \
  --false-alarm-rate 1e-7 2>&1 | tee /tmp/fasttracker_test.log | tail -50

echo ""

# Step 5: Results Verification
echo "=== Step 5/5: Results Verification ==="

# Check detection statistics
DETECTIONS=$(grep "Detections:" /tmp/fasttracker_test.log | awk '{print $2}')
CLUTTER=$(grep "Clutter:" /tmp/fasttracker_test.log | awk '{print $2}')

echo "Detection Statistics:"
echo "  - Target Detections: $DETECTIONS"
echo "  - Clutter Generated: $CLUTTER"

if [ "$DETECTIONS" -gt 0 ]; then
    echo "✅ Target detection working"
else
    echo "⚠️  Warning: No target detections (may be expected depending on scenario)"
fi

if [ "$CLUTTER" -gt 0 ]; then
    echo "✅ Clutter generation working"
else
    echo "⚠️  Warning: No clutter generated"
fi

echo ""
echo "Sample measurements.csv:"
docker exec fasttracker-fasttracker-1 cat /app/measurements.csv | head -10

echo ""
echo "========================================="
echo "✅ All Steps Completed Successfully"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Review test results above"
echo "  2. Run full scenario via Web GUI at http://localhost:5000"
echo "  3. Verify behavior matches expectations"
echo ""
