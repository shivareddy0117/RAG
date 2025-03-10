import os
import logging
import torch
import gc
from typing import List, Dict, Any, Optional, Union, Tuple
import time
from functools import wraps

import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPUs
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        "available": torch.cuda.is_available(),
        "count": torch.cuda.device_count(),
        "current_device": None,
        "current_device_name": None,
        "memory_allocated": None,
        "memory_reserved": None,
        "max_memory_allocated": None,
        "max_memory_reserved": None
    }
    
    if gpu_info["available"]:
        try:
            gpu_info["current_device"] = torch.cuda.current_device()
            gpu_info["current_device_name"] = torch.cuda.get_device_name(gpu_info["current_device"])
            gpu_info["memory_allocated"] = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            gpu_info["memory_reserved"] = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
            gpu_info["max_memory_allocated"] = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            gpu_info["max_memory_reserved"] = torch.cuda.max_memory_reserved() / (1024 ** 2)  # MB
        except Exception as e:
            logger.error(f"Error getting GPU info: {str(e)}")
    
    return gpu_info

def setup_gpu(use_gpu: bool = config.USE_GPU, 
             mixed_precision: bool = config.MIXED_PRECISION,
             cuda_visible_devices: Optional[str] = config.CUDA_VISIBLE_DEVICES) -> Dict[str, Any]:
    """
    Set up GPU and optimization settings
    
    Args:
        use_gpu: Whether to use GPU acceleration
        mixed_precision: Whether to use mixed precision
        cuda_visible_devices: Comma-separated list of GPU devices to use
        
    Returns:
        Dictionary with setup information
    """
    setup_info = {
        "using_gpu": False,
        "using_mixed_precision": False,
        "device": "cpu",
        "gpu_info": None
    }
    
    # Set CUDA devices before anything else
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    if use_gpu and torch.cuda.is_available():
        # Set default device to GPU
        setup_info["device"] = "cuda"
        setup_info["using_gpu"] = True
        
        # Get GPU info
        setup_info["gpu_info"] = get_gpu_info()
        
        # Enable mixed precision if requested
        if mixed_precision:
            torch.set_float32_matmul_precision('high')
            setup_info["using_mixed_precision"] = True
            logger.info("Enabled mixed precision training")
        
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Using CPU.")
        
        # Clear any existing CUDA environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    return setup_info

def clean_gpu_memory():
    """Free up GPU memory by clearing cache and garbage collection"""
    if torch.cuda.is_available():
        # Empty the cache and trigger garbage collection
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleaned GPU memory")

def benchmark_gpu(func):
    """
    Decorator to benchmark GPU performance of a function
    
    Args:
        func: Function to benchmark
        
    Returns:
        Wrapped function with benchmark
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Skip if CPU mode
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
        
        # Get starting memory usage
        start_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        start_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        
        # Track time
        start_time = time.time()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Calculate metrics
        end_time = time.time()
        end_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        end_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        
        # Log benchmark results
        logger.info(f"GPU Benchmark for {func.__name__}:")
        logger.info(f"  - Execution time: {end_time - start_time:.4f} seconds")
        logger.info(f"  - Memory allocated: {start_mem_allocated:.2f} MB -> {end_mem_allocated:.2f} MB (Δ {end_mem_allocated - start_mem_allocated:.2f} MB)")
        logger.info(f"  - Memory reserved: {start_mem_reserved:.2f} MB -> {end_mem_reserved:.2f} MB (Δ {end_mem_reserved - start_mem_reserved:.2f} MB)")
        
        return result
    
    return wrapper

def use_cuda_tensors(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Move tensors to CUDA if available
    
    Args:
        tensors: List of tensors to move
        
    Returns:
        List of tensors on the appropriate device
    """
    if torch.cuda.is_available():
        return [tensor.cuda() for tensor in tensors]
    return tensors

def optimize_model_for_inference(model):
    """
    Apply optimizations to a model for faster inference
    
    Args:
        model: PyTorch model to optimize
        
    Returns:
        Optimized model
    """
    if not torch.cuda.is_available():
        return model
    
    try:
        # Move model to CUDA
        model = model.cuda()
        
        # Set to evaluation mode
        model = model.eval()
        
        # Apply half-precision (fp16) if appropriate
        if config.MIXED_PRECISION:
            model = model.half()
        
        # Optional: could use torch.jit.trace or torch.jit.script for further optimization
        # Be careful with this as it might not work for all models
        # model = torch.jit.optimize_for_inference(torch.jit.script(model))
        
        logger.info("Model optimized for inference")
        
        return model
    except Exception as e:
        logger.warning(f"Could not fully optimize model for inference: {str(e)}")
        return model

def benchmark_inference(model, input_data, num_iterations: int = 10):
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model to benchmark
        input_data: Input data to use for benchmark
        num_iterations: Number of benchmark iterations
        
    Returns:
        Dictionary with benchmark results
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available for benchmarking"}
    
    try:
        # Ensure model is on GPU
        model = model.cuda()
        
        # Ensure input data is on GPU
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cuda()
        elif isinstance(input_data, list) and all(isinstance(item, torch.Tensor) for item in input_data):
            input_data = [item.cuda() for item in input_data]
        
        # Warmup
        with torch.no_grad():
            # Handle different input types
            if isinstance(input_data, torch.Tensor):
                _ = model(input_data)
            elif isinstance(input_data, dict):
                _ = model(**input_data)
            else:
                _ = model(input_data)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                if isinstance(input_data, torch.Tensor):
                    _ = model(input_data)
                elif isinstance(input_data, dict):
                    _ = model(**input_data)
                else:
                    _ = model(input_data)
                torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate results
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        
        return {
            "total_time_seconds": total_time,
            "avg_time_seconds": avg_time,
            "iterations_per_second": num_iterations / total_time,
            "num_iterations": num_iterations
        }
    except Exception as e:
        logger.error(f"Error during inference benchmark: {str(e)}")
        return {"error": str(e)} 