#!/usr/bin/env python
"""
Master script to run both layer profiling and pipeline execution.
This script profiles individual transformer layers and then evaluates pipeline execution.
"""

import os
import argparse
from layer_profiler import LayerProfiler
from pipeline_executor import PipelineExecutor
import pandas as pd
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Profile and evaluate transformer model pipelining")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                       help="Model to profile and evaluate (default: mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--kv-sizes", nargs="+", type=int, default=[64, 128, 256, 512],
                       help="KV cache sizes to test in MB (default: 64 128 256 512)")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of trials for each measurement (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for pipeline execution (default: 1)")
    parser.add_argument("--seq-len", type=int, default=64,
                       help="Sequence length for pipeline execution (default: 64)")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of samples to process in pipeline (default: 3)")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Directory to save results (default: ./results)")
    parser.add_argument("--skip-layer-profiling", action="store_true",
                       help="Skip layer profiling and only run pipeline execution")
    parser.add_argument("--skip-pipeline", action="store_true",
                       help="Skip pipeline execution and only run layer profiling")
    
    args = parser.parse_args()
    
    # Get model short name and create timestamp
    model_short_name = args.model.split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model-specific output directory with timestamp
    model_dir = os.path.join(args.output_dir, f"{model_short_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Running profiling for model: {args.model}")
    print(f"Results will be saved to: {model_dir}")
    
    # Save run configuration
    config = {
        "model": [args.model],
        "timestamp": [timestamp],
        "trials": [args.trials],
        "batch_size": [args.batch_size],
        "sequence_length": [args.seq_len],
        "samples": [args.samples],
        "skip_layer_profiling": [args.skip_layer_profiling],
        "skip_pipeline": [args.skip_pipeline],
    }
    pd.DataFrame(config).to_csv(f"{model_dir}/config.csv", index=False)
    
    layer_df = None
    kv_df = None
    
    # Run layer profiling if not skipped
    if not args.skip_layer_profiling:
        print("\n===== Running Layer Profiling =====")
        # Create and run profiler
        profiler = LayerProfiler(
            model_name=args.model,
            kv_cache_sizes_mb=args.kv_sizes,
            num_trials=args.trials,
        )
        
        # Profile layers
        profiler.profile_all_layers()
        
        # Profile KV cache transfer
        profiler.measure_kv_cache_transfer()
        
        # Generate report
        layer_df, kv_df = profiler.generate_report()
        
        # Plot results and save to model directory
        profiler.plot_results(model_dir)
        
        # Save results to CSV with model name in the filename
        layer_csv = f"{model_dir}/layer_profiling_{model_short_name}.csv"
        kv_csv = f"{model_dir}/kv_cache_profiling_{model_short_name}.csv"
        layer_df.to_csv(layer_csv, index=False)
        kv_df.to_csv(kv_csv, index=False)
        
        print(f"Layer profiling results saved to {model_dir}/")
    
    # Run pipeline execution if not skipped
    if not args.skip_pipeline:
        print("\n===== Running Pipeline Execution =====")
        # Create and run executor
        executor = PipelineExecutor(
            model_name=args.model,
            profiling_data_path=(f"{model_dir}/layer_profiling_{model_short_name}.csv" 
                                if layer_df is not None else None),
            cache_profiling_path=(f"{model_dir}/kv_cache_profiling_{model_short_name}.csv"
                                 if kv_df is not None else None),
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
        )
        
        # Run comparison
        seq_time, pipe_time, speedup = executor.run_comparison(num_samples=args.samples, output_dir=model_dir)

    print(f"\nAll profiling results saved to {model_dir}/")


if __name__ == "__main__":
    main() 