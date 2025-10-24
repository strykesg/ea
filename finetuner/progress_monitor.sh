#!/bin/bash

# Real-time training progress monitor with countdown
# Usage: ./progress_monitor.sh [log_file]

LOG_FILE="${1:-training.log}"
INTERVAL=10  # Update every 10 seconds

echo "🎯 DeepSeek-V2-Lite Training Progress Monitor"
echo "=============================================="
echo "Monitoring: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Function to calculate and display progress
show_progress() {
    # Extract latest epoch from log
    LATEST_EPOCH=$(tail -50 "$LOG_FILE" | grep -o "'epoch': [0-9.]*" | tail -1 | awk '{print $2}' | tr -d "'")

    if [ -z "$LATEST_EPOCH" ]; then
        echo "⏳ Waiting for training to start..."
        return
    fi

    # Calculate progress
    EPOCH_FLOAT=$(echo "$LATEST_EPOCH" | bc 2>/dev/null || echo "0")
    TOTAL_EPOCHS=9.0
    PROGRESS_PCT=$(echo "scale=1; $EPOCH_FLOAT / $TOTAL_EPOCHS * 100" | bc 2>/dev/null || echo "0")

    # Estimate completion time (assuming 4 hours total)
    if (( $(echo "$EPOCH_FLOAT > 0" | bc -l) )); then
        # Calculate rate: epochs per minute
        ELAPSED_MIN=$(tail -1 "$LOG_FILE" | cut -d' ' -f1-2 | date -f- +%s 2>/dev/null || echo "0")
        if [ "$ELAPSED_MIN" != "0" ]; then
            START_TIME=$(head -1 "$LOG_FILE" | cut -d' ' -f1-2 | date -f- +%s 2>/dev/null || echo "0")
            if [ "$START_TIME" != "0" ]; then
                ELAPSED_MIN=$(( (ELAPSED_MIN - START_TIME) / 60 ))
                RATE=$(echo "scale=4; $EPOCH_FLOAT / $ELAPSED_MIN" | bc 2>/dev/null || echo "0.02")
                REMAINING_EPOCHS=$(echo "scale=2; $TOTAL_EPOCHS - $EPOCH_FLOAT" | bc 2>/dev/null || echo "8.61")
                REMAINING_MIN=$(echo "scale=0; $REMAINING_EPOCHS / $RATE" | bc 2>/dev/null || echo "441")
                REMAINING_HOURS=$(echo "scale=1; $REMAINING_MIN / 60" | bc 2>/dev/null || echo "7.4")

                # Format remaining time
                if (( $(echo "$REMAINING_HOURS >= 1" | bc -l) )); then
                    TIME_REMAINING="${REMAINING_HOURS} hours"
                else
                    TIME_REMAINING="${REMAINING_MIN} minutes"
                fi
            else
                TIME_REMAINING="Calculating..."
            fi
        else
            TIME_REMAINING="Calculating..."
        fi
    else
        TIME_REMAINING="Calculating..."
    fi

    # Get latest loss and other metrics
    LATEST_LOSS=$(tail -20 "$LOG_FILE" | grep -o "'loss': [0-9.]*" | tail -1 | awk '{print $2}' | tr -d "'")
    LATEST_LR=$(tail -20 "$LOG_FILE" | grep -o "'learning_rate': [0-9e.-]*" | tail -1 | awk '{print $2}' | tr -d "'")

    # Clear screen and show progress
    echo -ne "\033[2J\033[H"  # Clear screen
    echo "🎯 DeepSeek-V2-Lite Training Progress Monitor"
    echo "=============================================="
    echo "📊 Current Epoch: ${EPOCH_FLOAT} / ${TOTAL_EPOCHS} (${PROGRESS_PCT}%)"
    echo "⏱️  Est. Time Remaining: ${TIME_REMAINING}"
    echo ""
    echo "📈 Training Metrics:"
    echo "   Loss: ${LATEST_LOSS:-'N/A'}"
    echo "   Learning Rate: ${LATEST_LR:-'N/A'}"
    echo ""
    echo "🎛️  Progress Bar:"

    # Create visual progress bar
    PROGRESS_INT=$(echo "scale=0; $PROGRESS_PCT / 1" | bc 2>/dev/null || echo "4")
    FILLED=$((PROGRESS_INT / 2))
    EMPTY=$((50 - FILLED))

    printf "   ["
    for ((i=1; i<=FILLED; i++)); do printf "█"; done
    for ((i=1; i<=EMPTY; i++)); do printf "░"; done
    printf "] %3.1f%%\n" "$PROGRESS_PCT"

    echo ""
    echo "📝 Last Update: $(date '+%H:%M:%S')"
    echo "💾 Log File: $LOG_FILE"
    echo ""
    echo "🔄 Next update in ${INTERVAL}s... (Ctrl+C to stop)"
}

# Main monitoring loop
while true; do
    if [ -f "$LOG_FILE" ]; then
        show_progress
    else
        echo "⏳ Waiting for log file: $LOG_FILE"
    fi
    sleep $INTERVAL
done
