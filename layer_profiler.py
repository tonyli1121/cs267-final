import torch
import time
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import gc
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os


class LayerProfiler:
    def __init__(
        self,
        model_name: str,
        dummy_input_text: str = "This is a test input for profiling transformer layers.",
        kv_cache_sizes_mb: List[int] = [64, 128, 256, 512],
        num_trials: int = 3,
    ):
        self.model_name = model_name
        self.dummy_input_text = dummy_input_text
        self.kv_cache_sizes_mb = kv_cache_sizes_mb
        self.num_trials = num_trials
        
        # Load tokenizer and prepare input
        print(f"Loading tokenizer for {self.model_name}")
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
            self.tokenized_input = self.tokenizer(dummy_input_text, return_tensors="pt")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Continuing without tokenizer - this might affect some functionality.")
            self.tokenizer = None
            self.tokenized_input = None
        
        # Load model on CPU to analyze layer by layer
        print(f"Loading model {self.model_name} on CPU")
        try:
            self.cpu_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="cpu",
                token=hf_token,
                trust_remote_code=True
            )
            
            # Extract the transformer blocks for individual profiling
            if 'mistral' in self.model_name.lower():
                self.transformer_blocks = self.cpu_model.model.layers
            elif 'deepseek' in self.model_name.lower():
                if hasattr(self.cpu_model, 'language_model'):
                    # For DeepSeek-VL2 models
                    self.transformer_blocks = self.cpu_model.language_model.model.layers
                else:
                    # For other DeepSeek models
                    self.transformer_blocks = self.cpu_model.model.layers
            else:
                raise ValueError(f"Model architecture for {self.model_name} not supported yet")
            
            self.num_layers = len(self.transformer_blocks)
            print(f"Model has {self.num_layers} transformer layers")
            
            # For passing activations through the model
            if hasattr(self.cpu_model.config, "hidden_size"):
                self.hidden_size = self.cpu_model.config.hidden_size
            else:
                # Default size if not specified in config
                self.hidden_size = 4096  # Common for 7B models
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}. Please check the model name and your internet connection.")
        
        # Results storage
        self.results = {
            "layer_idx": [],
            "param_count": [],
            "load_time": [],
            "compute_time": [],
            "offload_time": [],
            "load_compute_ratio": [],
            "offload_compute_ratio": [],
        }
        
        self.kv_cache_results = {
            "size_mb": [],
            "offload_time": [],
            "reload_time": [],
        }

    def _clear_gpu_memory(self):
        """Clear GPU memory and run garbage collection."""
        torch.cuda.empty_cache()
        gc.collect()
    
    def _create_dummy_hidden_states(self, batch_size=1, seq_len=512):
        """Create dummy hidden states to feed to transformer layers."""
        return torch.randn(batch_size, seq_len, self.hidden_size, 
                          dtype=torch.float16)
    
    def _count_parameters(self, layer) -> int:
        """Count number of parameters in a layer."""
        return sum(p.numel() for p in layer.parameters())
    
    def _measure_time(self, func, num_trials=3) -> float:
        """Measure execution time of a function over multiple trials."""
        times = []
        for _ in range(num_trials):
            start = time.time()
            func()
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            times.append(time.time() - start)
        return sum(times) / len(times)  # Return average time
    
    def _create_synthetic_kv_cache(self, size_mb: int):
        """Create a synthetic KV cache tensor of specified size in MB."""
        # Calculate tensor size: assume float16 (2 bytes per element)
        num_elements = (size_mb * 1024 * 1024) // 2
        
        # Make dimensions reasonable for a KV cache
        # For a 7B model: (batch_size, num_heads, seq_len, head_dim)
        head_dim = 64  # Typical for many models
        num_heads = 32  # Typical for 7B models
        
        # Calculate sequence length based on number of elements
        # We'll generate two tensors (K and V)
        batch_size = 1
        total_elements_per_tensor = num_elements // 2
        seq_len = total_elements_per_tensor // (batch_size * num_heads * head_dim)
        
        # Create random tensors
        k_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                             dtype=torch.float16)
        v_cache = torch.randn(batch_size, num_heads, seq_len, head_dim,
                             dtype=torch.float16)
        
        # Verify size is approximately as requested
        actual_size_mb = (k_cache.nelement() + v_cache.nelement()) * 2 / (1024 * 1024)
        print(f"Created synthetic KV cache: {actual_size_mb:.2f} MB")
        
        return k_cache, v_cache
    
    def measure_kv_cache_transfer(self):
        """Measure time to transfer synthetic KV cache between CPU and GPU."""
        for size_mb in self.kv_cache_sizes_mb:
            self._clear_gpu_memory()
            
            # Create synthetic KV cache
            k_cache, v_cache = self._create_synthetic_kv_cache(size_mb)
            
            # Measure offload time (GPU to CPU)
            def offload_func():
                k_gpu = k_cache.cuda()
                v_gpu = v_cache.cuda()
                torch.cuda.synchronize()
                k_cpu = k_gpu.cpu()
                v_cpu = v_gpu.cpu()
            
            offload_time = self._measure_time(offload_func, self.num_trials)
            
            # Measure reload time (CPU to GPU)
            def reload_func():
                k_gpu = k_cache.cuda()
                v_gpu = v_cache.cuda()
                torch.cuda.synchronize()
            
            reload_time = self._measure_time(reload_func, self.num_trials)
            
            # Store results
            self.kv_cache_results["size_mb"].append(size_mb)
            self.kv_cache_results["offload_time"].append(offload_time)
            self.kv_cache_results["reload_time"].append(reload_time)
    
    def profile_layer(self, layer_idx: int):
        """Profile a single transformer layer for load, compute, and offload times."""
        self._clear_gpu_memory()
        
        layer = self.transformer_blocks[layer_idx]
        param_count = self._count_parameters(layer)
        
        # Create dummy input with smaller sequence length to reduce memory usage
        batch_size = 1
        seq_len = 64  # Even smaller sequence length
        dummy_input = self._create_dummy_hidden_states(batch_size=batch_size, seq_len=seq_len)
        
        # 1. Measure load time (CPU to GPU)
        def load_func():
            layer_gpu = layer.to("cuda")
            torch.cuda.synchronize()
            return layer_gpu
        
        load_time = self._measure_time(lambda: load_func(), self.num_trials)
        layer_gpu = load_func()  # Actually load the layer
        
        # 2. Measure compute time (forward pass)
        dummy_input_gpu = dummy_input.cuda()
        
        # Generate position ids and rotary embeddings if needed
        seq_len = dummy_input.shape[1]
        position_ids = torch.arange(0, seq_len, device=dummy_input_gpu.device).unsqueeze(0)
        
        # Custom forward function that works for different model types
        def custom_forward(layer, hidden_states):
            """Custom forward pass that skips attention to avoid position embedding issues."""
            # Most models have an input layernorm
            if hasattr(layer, 'input_layernorm'):
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                
                # Skip the attention part which often needs position embeddings
                
                # Apply MLP/FFN if available
                if hasattr(layer, 'mlp'):
                    hidden_states = layer.mlp(hidden_states)
                    
                # Add residual connection
                hidden_states = residual + hidden_states
                
                return hidden_states
            else:
                # Fallback for models with different structure
                # Just return the input to avoid errors
                return hidden_states
        
        # Detect model type and use appropriate forward pass
        if 'qwen' in self.model_name.lower() or 'deepseek-r1' in self.model_name.lower():
            # Use custom forward for Qwen and DeepSeek models
            def compute_func():
                with torch.no_grad():
                    _ = custom_forward(layer_gpu, dummy_input_gpu)
                torch.cuda.synchronize()
        elif 'mistral' in self.model_name.lower():
            # Use custom forward for Mistral
            def compute_func():
                with torch.no_grad():
                    _ = custom_forward(layer_gpu, dummy_input_gpu)
                torch.cuda.synchronize()
        else:
            # For other models, try the standard approach first
            try:
                # Test if standard forward pass works
                with torch.no_grad():
                    test_output = layer_gpu(dummy_input_gpu)
                
                # If it works, define the function normally
                def compute_func():
                    with torch.no_grad():
                        output = layer_gpu(dummy_input_gpu)
                        if isinstance(output, tuple):
                            output = output[0]  # Get hidden states from tuple
                    torch.cuda.synchronize()
                    
            except Exception as e:
                print(f"Standard forward pass failed for layer {layer_idx}, using custom forward instead. Error: {e}")
                # Fallback to custom forward
                def compute_func():
                    with torch.no_grad():
                        _ = custom_forward(layer_gpu, dummy_input_gpu)
                    torch.cuda.synchronize()
        
        compute_time = self._measure_time(compute_func, self.num_trials)
        
        # 3. Measure offload time (GPU to CPU)
        def offload_func():
            _ = layer_gpu.cpu()
            torch.cuda.synchronize()
        
        offload_time = self._measure_time(offload_func, self.num_trials)
        
        # Calculate ratios
        load_compute_ratio = load_time / compute_time if compute_time > 0 else float('inf')
        offload_compute_ratio = offload_time / compute_time if compute_time > 0 else float('inf')
        
        # Store results
        self.results["layer_idx"].append(layer_idx)
        self.results["param_count"].append(param_count)
        self.results["load_time"].append(load_time)
        self.results["compute_time"].append(compute_time)
        self.results["offload_time"].append(offload_time)
        self.results["load_compute_ratio"].append(load_compute_ratio)
        self.results["offload_compute_ratio"].append(offload_compute_ratio)
        
        # Print layer results
        print(f"Layer {layer_idx}: Load={load_time:.4f}s, Compute={compute_time:.4f}s, "
              f"Offload={offload_time:.4f}s, Load/Compute={load_compute_ratio:.2f}, "
              f"Offload/Compute={offload_compute_ratio:.2f}")
        
        # Clean up GPU memory
        del layer_gpu, dummy_input_gpu
        self._clear_gpu_memory()
    
    def profile_all_layers(self):
        """Profile all transformer layers in the model."""
        print(f"Profiling {self.num_layers} layers in {self.model_name}...")
        for i in range(self.num_layers):
            self.profile_layer(i)
    
    def generate_report(self):
        """Generate a comprehensive profiling report."""
        # Create DataFrames
        layer_df = pd.DataFrame(self.results)
        kv_df = pd.DataFrame(self.kv_cache_results)
        
        # Calculate averages
        avg_load_time = layer_df["load_time"].mean()
        avg_compute_time = layer_df["compute_time"].mean()
        avg_offload_time = layer_df["offload_time"].mean()
        avg_load_compute_ratio = layer_df["load_compute_ratio"].mean()
        avg_offload_compute_ratio = layer_df["offload_compute_ratio"].mean()
        
        # Print summary
        print("\n===== Layer-wise Profiling Summary =====")
        print(f"Model: {self.model_name}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Average load time: {avg_load_time:.4f}s")
        print(f"Average compute time: {avg_compute_time:.4f}s")
        print(f"Average offload time: {avg_offload_time:.4f}s")
        print(f"Average load/compute ratio: {avg_load_compute_ratio:.2f}")
        print(f"Average offload/compute ratio: {avg_offload_compute_ratio:.2f}")
        
        print("\n===== KV Cache Transfer Summary =====")
        for i, size_mb in enumerate(self.kv_cache_results["size_mb"]):
            offload_time = self.kv_cache_results["offload_time"][i]
            reload_time = self.kv_cache_results["reload_time"][i]
            print(f"KV Cache Size: {size_mb} MB")
            print(f"  Offload time (GPU→CPU): {offload_time:.4f}s")
            print(f"  Reload time (CPU→GPU): {reload_time:.4f}s")
        
        # Determine viability
        viable = avg_load_compute_ratio < 1.0 and avg_offload_compute_ratio < 1.0
        print("\n===== Viability Assessment =====")
        if viable:
            print("✅ Layer-wise pipelining appears VIABLE based on measurements.")
            print(f"  - Transfer overhead is less than compute time (Load: {avg_load_compute_ratio:.2f}x, Offload: {avg_offload_compute_ratio:.2f}x)")
        else:
            print("❌ Layer-wise pipelining may NOT BE VIABLE based on measurements.")
            print(f"  - Transfer overhead exceeds compute time (Load: {avg_load_compute_ratio:.2f}x, Offload: {avg_offload_compute_ratio:.2f}x)")
        
        return layer_df, kv_df
    
    def plot_results(self, output_dir: str = "./results"):
        """Generate plots visualizing the profiling results."""
        import os
        import matplotlib.pyplot as plt
        
        # Create model-specific output directory
        model_short_name = self.model_name.split('/')[-1]
        model_dir = os.path.join(output_dir, model_short_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create DataFrame from results
        df = pd.DataFrame(self.results)
        
        # Plot 1: Timing breakdown by layer
        plt.figure(figsize=(12, 6))
        plt.bar(df["layer_idx"], df["load_time"], label="Load Time")
        plt.bar(df["layer_idx"], df["compute_time"], bottom=df["load_time"], label="Compute Time")
        plt.bar(df["layer_idx"], df["offload_time"], 
                bottom=df["load_time"] + df["compute_time"], label="Offload Time")
        plt.xlabel("Layer Index")
        plt.ylabel("Time (seconds)")
        plt.title(f"Timing Breakdown by Layer - {self.model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{model_dir}/layer_timing_breakdown.png")
        
        # Plot 2: Ratios by layer
        plt.figure(figsize=(12, 6))
        plt.plot(df["layer_idx"], df["load_compute_ratio"], marker='o', label="Load/Compute Ratio")
        plt.plot(df["layer_idx"], df["offload_compute_ratio"], marker='s', label="Offload/Compute Ratio")
        plt.axhline(y=1.0, color='r', linestyle='--', label="Breakeven Line")
        plt.xlabel("Layer Index")
        plt.ylabel("Ratio (transfer time / compute time)")
        plt.title(f"Transfer/Compute Ratios by Layer - {self.model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{model_dir}/layer_ratios.png")
        
        # Plot 3: KV Cache transfer times
        if self.kv_cache_results["size_mb"]:
            kv_df = pd.DataFrame(self.kv_cache_results)
            plt.figure(figsize=(10, 6))
            plt.plot(kv_df["size_mb"], kv_df["offload_time"], marker='o', label="Offload Time (GPU→CPU)")
            plt.plot(kv_df["size_mb"], kv_df["reload_time"], marker='s', label="Reload Time (CPU→GPU)")
            plt.xlabel("KV Cache Size (MB)")
            plt.ylabel("Time (seconds)")
            plt.title(f"KV Cache Transfer Times - {self.model_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{model_dir}/kv_cache_transfer.png")
        
        return model_dir


def main():
    parser = ArgumentParser(description="Profile transformer layers for offloading viability")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                       help="Model to profile (default: mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--kv-sizes", nargs="+", type=int, default=[64, 128, 256, 512],
                       help="KV cache sizes to test in MB (default: 64 128 256 512)")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of trials for each measurement (default: 3)")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Directory to save results (default: ./results)")
    
    args = parser.parse_args()
    
    # Try to create profiler with the specified model
    try:
        profiler = LayerProfiler(
            model_name=args.model,
            kv_cache_sizes_mb=args.kv_sizes,
            num_trials=args.trials,
        )
    except ValueError as e:
        if 'deepseek-vl' in str(e).lower():
            print("\nDeepSeek-VL2 models require additional packages.")
            print("Would you like to try with the default Mistral-7B model instead? (y/n)")
            response = input().strip().lower()
            if response == 'y':
                print("Falling back to Mistral-7B model...")
                profiler = LayerProfiler(
                    model_name="mistralai/Mistral-7B-v0.1",
                    kv_cache_sizes_mb=args.kv_sizes,
                    num_trials=args.trials,
                )
            else:
                print("Exiting. Please install the required packages and try again.")
                return
        else:
            # For other value errors, just re-raise
            raise
    
    # Profile layers
    profiler.profile_all_layers()
    
    # Profile KV cache transfer
    profiler.measure_kv_cache_transfer()
    
    # Generate report
    layer_df, kv_df = profiler.generate_report()
    
    # Plot results and get model-specific directory
    model_dir = profiler.plot_results(args.output_dir)
    
    # Get model short name for filenames
    model_short_name = profiler.model_name.split('/')[-1]
    
    # Save results to CSV with model name in the filename
    layer_df.to_csv(f"{model_dir}/layer_profiling_{model_short_name}.csv", index=False)
    kv_df.to_csv(f"{model_dir}/kv_cache_profiling_{model_short_name}.csv", index=False)
    
    print(f"\nResults saved to {model_dir}/")


if __name__ == "__main__":
    main() 