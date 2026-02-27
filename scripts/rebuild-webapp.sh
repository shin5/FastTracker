#!/bin/bash
# FastTracker: Quick Webapp Rebuild (No C++ compilation)
# Use this for Python/HTML/JavaScript changes only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================="
echo "FastTracker: Webapp Quick Rebuild"
echo "========================================="
echo ""

# Step 1: Rebuild (no --no-cache for speed)
echo "=== Step 1/3: Docker Rebuild (webapp only) ==="
docker compose build 2>&1 | tail -10
echo "✅ Build complete"
echo ""

# Step 2: Restart Container
echo "=== Step 2/3: Container Restart ==="
docker compose down
sleep 1
docker compose up -d
sleep 2
echo "✅ Container restarted"
echo ""

# Step 3: Verify Template Update
echo "=== Step 3/3: Verify Template Update ==="
TEMPLATE_TIME=$(docker exec fasttracker-fasttracker-1 stat -c '%y' /app/python/webapp/templates/index.html 2>/dev/null || echo "ERROR")

if [ "$TEMPLATE_TIME" = "ERROR" ]; then
    echo "❌ Could not verify template (container may not be ready)"
else
    echo "Template timestamp: $TEMPLATE_TIME"

    # Check if timestamp is recent (within last 5 minutes)
    TEMPLATE_EPOCH=$(docker exec fasttracker-fasttracker-1 stat -c '%Y' /app/python/webapp/templates/index.html)
    CURRENT_EPOCH=$(date +%s)
    TIME_DIFF=$((CURRENT_EPOCH - TEMPLATE_EPOCH))

    if [ $TIME_DIFF -lt 300 ]; then
        echo "✅ Template is recent (${TIME_DIFF}s old)"
    else
        echo "⚠️  Warning: Template is ${TIME_DIFF}s old (may not be updated)"
    fi
fi

echo ""
echo "========================================="
echo "✅ Webapp Update Complete"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:5000"
echo "  2. Hard refresh browser:"
echo "     - Windows/Linux: Ctrl + Shift + R"
echo "     - Mac: Cmd + Shift + R"
echo "  3. Or open in incognito/private mode"
echo "  4. Verify changes appear in UI"
echo ""
