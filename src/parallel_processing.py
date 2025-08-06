#!/usr/bin/env python3
"""
Parallel Processing Module for JWST Photometry Pipeline
Part of Phase 7: Performance and Scalability

This module provides parallel processing capabilities for multi-band operations
and large dataset processing optimization.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import functools

# Setup logging
logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Parallel processing manager for JWST photometry operations."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = True,
                 chunk_size: int = 1000000):  # 1M pixels per chunk
        """
        Initialize parallel processor.
        
        Parameters:
        -----------
        max_workers : int, optional
            Maximum number of worker processes/threads
        use_processes : bool
            Use processes (True) or threads (False)
        chunk_size : int
            Size of image chunks for processing
        """
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        
        logger.info(f"Initialized ParallelProcessor: {self.max_workers} workers, "
                   f"{'processes' if use_processes else 'threads'}")
    
    def process_bands_parallel(self, 
                             band_files: Dict[str, Path],
                             processing_function: Callable,
                             **kwargs) -> Dict[str, Any]:
        """
        Process multiple bands in parallel.
        
        Parameters:
        -----------
        band_files : dict
            Dictionary mapping band names to file paths
        processing_function : callable
            Function to process each band
        **kwargs : dict
            Additional arguments for processing function
            
        Returns:
        --------
        dict : Results for each band
        """
        logger.info(f"Starting parallel processing of {len(band_files)} bands")
        start_time = time.time()
        
        # Prepare tasks
        tasks = []
        for band_name, file_path in band_files.items():
            task = functools.partial(processing_function, 
                                   file_path=file_path, 
                                   band_name=band_name, 
                                   **kwargs)
            tasks.append((band_name, task))
        
        # Execute in parallel
        results = {}
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_band = {
                executor.submit(task): band_name 
                for band_name, task in tasks
            }
            
            # Collect results
            for future in future_to_band:
                band_name = future_to_band[future]
                try:
                    result = future.result()
                    results[band_name] = result
                    logger.info(f"✅ Completed processing {band_name}")
                except Exception as e:
                    logger.error(f"❌ Failed processing {band_name}: {e}")
                    results[band_name] = None
        
        elapsed_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def chunk_large_image(self, image: np.ndarray, 
                         overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split large image into overlapping chunks for processing.
        
        Parameters:
        -----------
        image : np.ndarray
            Large image array
        overlap : int
            Overlap between chunks in pixels
            
        Returns:
        --------
        list : List of chunk dictionaries with slice information
        """
        height, width = image.shape
        
        # Calculate optimal chunk dimensions
        total_pixels = height * width
        chunk_side = int(np.sqrt(self.chunk_size))
        
        chunks = []
        
        y_start = 0
        while y_start < height:
            y_end = min(y_start + chunk_side, height)
            
            x_start = 0
            while x_start < width:
                x_end = min(x_start + chunk_side, width)
                
                # Add overlap except at edges
                y_slice_start = max(0, y_start - overlap)
                y_slice_end = min(height, y_end + overlap)
                x_slice_start = max(0, x_start - overlap)
                x_slice_end = min(width, x_end + overlap)
                
                chunk_info = {
                    'chunk_id': len(chunks),
                    'slice': (slice(y_slice_start, y_slice_end), 
                             slice(x_slice_start, x_slice_end)),
                    'processing_region': (slice(y_start - y_slice_start, 
                                               y_end - y_slice_start),
                                        slice(x_start - x_slice_start, 
                                             x_end - x_slice_start)),
                    'output_region': (slice(y_start, y_end), 
                                    slice(x_start, x_end)),
                    'data': image[y_slice_start:y_slice_end, 
                                x_slice_start:x_slice_end]
                }
                
                chunks.append(chunk_info)
                
                x_start += chunk_side
            
            y_start += chunk_side
        
        logger.info(f"Split {height}x{width} image into {len(chunks)} chunks")
        return chunks
    
    def process_chunks_parallel(self, 
                              chunks: List[Dict[str, Any]],
                              processing_function: Callable,
                              **kwargs) -> List[Any]:
        """
        Process image chunks in parallel.
        
        Parameters:
        -----------
        chunks : list
            List of chunk dictionaries
        processing_function : callable
            Function to process each chunk
        **kwargs : dict
            Additional arguments for processing function
            
        Returns:
        --------
        list : Results for each chunk
        """
        logger.info(f"Processing {len(chunks)} chunks in parallel")
        start_time = time.time()
        
        # Prepare tasks
        tasks = []
        for chunk in chunks:
            task = functools.partial(processing_function, 
                                   chunk=chunk, 
                                   **kwargs)
            tasks.append(task)
        
        # Execute in parallel
        results = [None] * len(chunks)
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(task): i 
                for i, task in enumerate(tasks)
            }
            
            # Collect results
            for future in future_to_index:
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"❌ Failed processing chunk {index}: {e}")
                    results[index] = None
        
        elapsed_time = time.time() - start_time
        logger.info(f"Chunk processing completed in {elapsed_time:.2f} seconds")
        
        return results

class MemoryOptimizer:
    """Memory optimization utilities for large dataset processing."""
    
    @staticmethod
    def estimate_memory_usage(image_shape: tuple, 
                            n_bands: int = 3,
                            dtype: np.dtype = np.float32) -> float:
        """
        Estimate memory usage for processing.
        
        Parameters:
        -----------
        image_shape : tuple
            Shape of individual images
        n_bands : int
            Number of bands to process
        dtype : np.dtype
            Data type of arrays
            
        Returns:
        --------
        float : Estimated memory usage in GB
        """
        bytes_per_pixel = np.dtype(dtype).itemsize
        pixels_per_image = np.prod(image_shape)
        
        # Estimate total memory including temporary arrays
        base_memory = n_bands * pixels_per_image * bytes_per_pixel
        working_memory = base_memory * 2  # For intermediate calculations
        
        total_gb = (base_memory + working_memory) / (1024**3)
        
        return total_gb
    
    @staticmethod
    def get_optimal_chunk_size(available_memory_gb: float,
                             image_shape: tuple,
                             safety_factor: float = 0.5) -> int:
        """
        Calculate optimal chunk size based on available memory.
        
        Parameters:
        -----------
        available_memory_gb : float
            Available memory in GB
        image_shape : tuple
            Shape of images to process
        safety_factor : float
            Safety factor for memory usage
            
        Returns:
        --------
        int : Optimal chunk size in pixels
        """
        usable_memory = available_memory_gb * safety_factor
        bytes_per_pixel = 4  # Assume float32
        
        max_pixels = int(usable_memory * (1024**3) / bytes_per_pixel)
        
        return max_pixels

def demonstrate_parallel_processing():
    """Demonstrate parallel processing capabilities."""
    
    logger.info("=== PARALLEL PROCESSING DEMONSTRATION ===")
    
    # Create sample data
    logger.info("Creating sample multi-band data...")
    sample_bands = {}
    for band in ['F150W', 'F277W', 'F444W']:
        # Simulate small images for demo
        sample_bands[band] = np.random.normal(1000, 100, (1000, 1000))
    
    # Simple processing function for demo
    def process_single_band(data, band_name):
        """Simple processing function for demonstration."""
        logger.info(f"Processing {band_name} with shape {data.shape}")
        time.sleep(1)  # Simulate processing time
        
        # Simple operations
        background = np.median(data)
        noise = np.std(data)
        n_sources = np.sum(data > background + 5*noise)
        
        return {
            'band': band_name,
            'background': background,
            'noise': noise,
            'sources': n_sources,
            'shape': data.shape
        }
    
    # Test sequential processing
    logger.info("\n--- Sequential Processing ---")
    start_time = time.time()
    sequential_results = {}
    for band_name, data in sample_bands.items():
        sequential_results[band_name] = process_single_band(data, band_name)
    sequential_time = time.time() - start_time
    
    logger.info(f"Sequential processing time: {sequential_time:.2f} seconds")
    
    # Test parallel processing
    logger.info("\n--- Parallel Processing ---")
    processor = ParallelProcessor(max_workers=3, use_processes=False)
    
    start_time = time.time()
    
    # Adapt function for parallel execution
    def parallel_task(file_path=None, band_name=None, data=None):
        if data is None:
            data = sample_bands[band_name]
        return process_single_band(data, band_name)
    
    # Create dummy file paths for demo
    band_files = {band: f"{band}.fits" for band in sample_bands.keys()}
    
    # Execute parallel processing
    parallel_results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(parallel_task, band_name=band, 
                          data=sample_bands[band]): band 
            for band in sample_bands.keys()
        }
        
        for future in futures:
            band = futures[future]
            parallel_results[band] = future.result()
    
    parallel_time = time.time() - start_time
    
    logger.info(f"Parallel processing time: {parallel_time:.2f} seconds")
    logger.info(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Demonstrate memory optimization
    logger.info("\n--- Memory Optimization Analysis ---")
    optimizer = MemoryOptimizer()
    
    # Typical JWST image
    jwst_shape = (24910, 19200)
    estimated_memory = optimizer.estimate_memory_usage(jwst_shape, n_bands=3)
    logger.info(f"Estimated memory for full JWST field: {estimated_memory:.2f} GB")
    
    # Optimal chunk size for different memory scenarios
    for available_memory in [4, 8, 16, 32]:
        chunk_size = optimizer.get_optimal_chunk_size(available_memory, jwst_shape)
        chunk_side = int(np.sqrt(chunk_size))
        logger.info(f"{available_memory} GB RAM → {chunk_side}x{chunk_side} pixel chunks")
    
    logger.info("\n=== DEMONSTRATION COMPLETE ===")
    logger.info("✅ Parallel processing implementation ready")
    logger.info("✅ Memory optimization tools available")
    logger.info("⏳ Integration with main pipeline needed")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    demonstrate_parallel_processing()
