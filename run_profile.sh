#!/bin/bash

# Simple shell script to run model profiling with common configurations

# Default values
MODEL="mistralai/Mistral-7B-v0.1"
MODE="full" # Options: full, layers, pipeline
KV_SIZES="64 128 256 512"
TRIALS=3
BATCH_SIZE=1
SEQ_LEN=64
SAMPLES=3
OUTPUT_DIR="./results"

# Display usage information
function show_usage {
    echo "Usage: ./run_profile.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --model MODEL       Model to profile (default: $MODEL)"
    echo "  --mode MODE         Profiling mode: full, layers, pipeline (default: $MODE)"
    echo "  --kv-sizes SIZES    KV cache sizes to test in MB (default: $KV_SIZES)"
    echo "  --trials N          Number of trials for each measurement (default: $TRIALS)"
    echo "  --batch-size N      Batch size for pipeline execution (default: $BATCH_SIZE)"
    echo "  --seq-len N         Sequence length for pipeline execution (default: $SEQ_LEN)"
    echo "  --samples N         Number of samples to process in pipeline (default: $SAMPLES)"
    echo "  --output-dir DIR    Directory to save results (default: $OUTPUT_DIR)"
    echo "  --help              Show this help message"
    echo
    echo "Example:"
    echo "  ./run_profile.sh --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --mode layers --trials 5"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --kv-sizes)
            KV_SIZES="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Convert KV_SIZES string to array format for Python
KV_SIZES_ARG=$(echo $KV_SIZES | tr ' ' ' ')

# Set up command based on mode
SKIP_LAYERS=""
SKIP_PIPELINE=""

case "$MODE" in
    "layers")
        SKIP_PIPELINE="--skip-pipeline"
        ;;
    "pipeline")
        SKIP_LAYERS="--skip-layer-profiling"
        ;;
    "full")
        # No skip flags needed
        ;;
    *)
        echo "Invalid mode: $MODE. Valid options are: full, layers, pipeline"
        exit 1
        ;;
esac

# Print configuration
echo "================================"
echo "Running profiling with settings:"
echo "--------------------------------"
echo "Model:        $MODEL"
echo "Mode:         $MODE"
echo "KV Sizes:     $KV_SIZES"
echo "Trials:       $TRIALS"
echo "Batch Size:   $BATCH_SIZE"
echo "Seq Length:   $SEQ_LEN"
echo "Samples:      $SAMPLES"
echo "Output Dir:   $OUTPUT_DIR"
echo "================================"

# Run the profiling script
CMD="python run_profiling.py --model $MODEL --kv-sizes $KV_SIZES_ARG --trials $TRIALS --batch-size $BATCH_SIZE --seq-len $SEQ_LEN --samples $SAMPLES --output-dir $OUTPUT_DIR $SKIP_LAYERS $SKIP_PIPELINE"

echo "Running command: $CMD"
echo "--------------------------------"

# Execute the command
$CMD

# Get the exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "--------------------------------"
    echo "✅ Profiling completed successfully!"
else
    echo "--------------------------------"
    echo "❌ Profiling failed with exit code $EXIT_CODE"
fi 