#!/bin/bash
# MLXLayerStream — Device Benchmark Automation
# Adapted from H2OAttnScore/run_device_benchmark.sh
#
# Usage:
#   ./run_device_benchmark.sh                    # Build + run default model
#   ./run_device_benchmark.sh MODEL_DIR          # Run specific model
#   SKIP_BUILD=1 ./run_device_benchmark.sh       # Skip build step
#
# Tested models (fit in 8GB device memory):
#   Qwen3.5-0.8B-8bit, Qwen3.5-2B-6bit, Qwen3.5-4B-4bit, Qwen3.5-9B-6bit

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$SCRIPT_DIR/StreamBenchmarkApp"
BUNDLE_ID="com.atomgradient.StreamBenchmarkApp"
MODEL_DIR="${1:-/Users/alex/Documents/mlx-community/Qwen3.5-4B-4bit}"
MODEL_NAME="$(basename "$MODEL_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results/device"
WAIT_SECONDS="${WAIT_SECONDS:-900}"

echo "=== MLXLayerStream Device Benchmark ==="
echo "Model: $MODEL_NAME ($MODEL_DIR)"
echo ""

# ── Build ──────────────────────────────────────────────────────────
if [ -z "$SKIP_BUILD" ]; then
    echo "--- Building StreamBenchmarkApp ---"
    cd "$APP_DIR"

    # Generate Xcode project from project.yml
    if command -v xcodegen &> /dev/null; then
        xcodegen generate 2>&1 | tail -1
    fi

    xcodebuild build \
        -project StreamBenchmarkApp.xcodeproj \
        -scheme StreamBenchmarkApp \
        -configuration Release \
        -destination 'generic/platform=iOS' \
        -allowProvisioningUpdates \
        -quiet 2>&1 || {
        echo "Build failed, trying clean build..."
        xcodebuild clean build \
            -project StreamBenchmarkApp.xcodeproj \
            -scheme StreamBenchmarkApp \
            -configuration Release \
            -destination 'generic/platform=iOS' \
            -allowProvisioningUpdates \
            -quiet 2>&1
    }
    echo "Build complete."
    cd "$SCRIPT_DIR"
else
    echo "Skipping build (SKIP_BUILD=1)"
fi

# Find built app
APP_PATH=$(find ~/Library/Developer/Xcode/DerivedData/StreamBenchmarkApp-*/Build/Products/*-iphoneos -name "StreamBenchmarkApp.app" -maxdepth 1 2>/dev/null | head -1)
if [ -z "$APP_PATH" ]; then
    echo "ERROR: Built app not found. Run without SKIP_BUILD."
    exit 1
fi
echo "App: $APP_PATH"

# ── Discover Devices ───────────────────────────────────────────────
echo ""
echo "--- Discovering devices ---"
DEVICES=$(xcrun devicectl list devices 2>/dev/null | grep -E "connected" | grep -v "Simulator")

if [ -z "$DEVICES" ]; then
    echo "No connected iOS devices found."
    exit 1
fi

echo "$DEVICES" | while IFS= read -r line; do
    NAME=$(echo "$line" | awk -F'  +' '{print $1}' | xargs)
    echo "  Found: $NAME"
done

mkdir -p "$RESULTS_DIR"

# ── Run Per Device ─────────────────────────────────────────────────
echo "$DEVICES" | while IFS= read -r line; do
    NAME=$(echo "$line" | awk -F'  +' '{print $1}' | xargs)
    UUID=$(echo "$line" | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}')

    if [ -z "$UUID" ]; then
        echo "Could not parse UUID for device: $NAME"
        continue
    fi

    echo ""
    echo "=== Device: $NAME ($UUID) ==="

    # Install app
    echo "Installing app..."
    xcrun devicectl device install app --device "$UUID" "$APP_PATH" 2>&1 | tail -2

    # Kill existing instance
    xcrun devicectl device process terminate --device "$UUID" "$BUNDLE_ID" 2>/dev/null || true
    sleep 2

    # Push model files
    echo "Pushing model: $MODEL_NAME..."

    for f in "$MODEL_DIR"/*.json "$MODEL_DIR"/*.safetensors "$MODEL_DIR"/tokenizer*; do
        if [ -f "$f" ]; then
            FNAME="$(basename "$f")"
            echo "  Copying $FNAME..."
            xcrun devicectl device copy to \
                --device "$UUID" \
                --source "$f" \
                --destination "Documents/model/$FNAME" \
                --domain-type appDataContainer \
                --domain-identifier "$BUNDLE_ID" 2>/dev/null || echo "    (warning: failed to push $FNAME)"
        fi
    done

    # Write model name
    echo "$MODEL_NAME" > /tmp/model_name.txt
    xcrun devicectl device copy to \
        --device "$UUID" \
        --source /tmp/model_name.txt \
        --destination "Documents/model_name.txt" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" 2>/dev/null

    # Clear previous results
    echo "" > /tmp/benchmark_results.txt
    xcrun devicectl device copy to \
        --device "$UUID" \
        --source /tmp/benchmark_results.txt \
        --destination "Documents/benchmark_results.txt" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" 2>/dev/null || true

    # Launch app
    echo "Launching benchmark..."
    xcrun devicectl device process launch --device "$UUID" "$BUNDLE_ID" 2>&1 | tail -1

    # Poll for results
    echo "Waiting for results (timeout: ${WAIT_SECONDS}s)..."
    ELAPSED=0
    INTERVAL=10
    RESULT_FILE="$RESULTS_DIR/${NAME// /_}_${MODEL_NAME}.txt"

    while [ $ELAPSED -lt $WAIT_SECONDS ]; do
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))

        # Try to pull results
        xcrun devicectl device copy from \
            --device "$UUID" \
            --source "Documents/benchmark_results.txt" \
            --destination /tmp/device_results.txt \
            --domain-type appDataContainer \
            --domain-identifier "$BUNDLE_ID" 2>/dev/null || continue

        if grep -q "=== END ===" /tmp/device_results.txt 2>/dev/null; then
            echo "Results received! (${ELAPSED}s)"
            cp /tmp/device_results.txt "$RESULT_FILE"
            echo ""
            cat "$RESULT_FILE"
            break
        fi

        # Auto-detect error reports (don't wait full timeout on failures)
        if grep -q "=== END ERROR REPORT ===" /tmp/device_results.txt 2>/dev/null; then
            echo "ERROR detected on device! (${ELAPSED}s)"
            cp /tmp/device_results.txt "$RESULT_FILE"
            echo ""
            cat "$RESULT_FILE"
            break
        fi

        echo "  ... waiting (${ELAPSED}s)"
    done

    if [ $ELAPSED -ge $WAIT_SECONDS ]; then
        echo "TIMEOUT after ${WAIT_SECONDS}s"
        # Try to get partial results
        xcrun devicectl device copy from \
            --device "$UUID" \
            --source "Documents/benchmark_results.txt" \
            --destination "$RESULT_FILE" \
            --domain-type appDataContainer \
            --domain-identifier "$BUNDLE_ID" 2>/dev/null || true
    fi

    # Pull detailed log file for diagnostics
    LOG_FILE="$RESULTS_DIR/${NAME// /_}_${MODEL_NAME}_log.txt"
    xcrun devicectl device copy from \
        --device "$UUID" \
        --source "Documents/benchmark_log.txt" \
        --destination "$LOG_FILE" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" 2>/dev/null && echo "Log saved: $LOG_FILE" || true

    # Kill app
    xcrun devicectl device process terminate --device "$UUID" "$BUNDLE_ID" 2>/dev/null || true
done

echo ""
echo "=== All results saved to $RESULTS_DIR ==="
ls -la "$RESULTS_DIR"/ 2>/dev/null
