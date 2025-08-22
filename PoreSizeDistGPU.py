#!/usr/bin/env python3
# GPU-Accelerated Pore Size Distribution Analysis
# Uses CuPy for GPU computation with automatic CPU fallback

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time

# GPU availability check
try:
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndi
    from cupyx.scipy import ndimage as gpu_ndimage
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration available!")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy not found. Install with: pip install cupy")
    print("   Falling back to CPU processing...")

# CPU fallback imports
import scipy.ndimage as ndi
from skimage import morphology, segmentation

def check_gpu_memory(required_gb=None):
    """Check available GPU memory"""
    if not GPU_AVAILABLE:
        return False, "No GPU available"
    
    try:
        # Get GPU memory info
        mempool = cp.get_default_memory_pool()
        device = cp.cuda.Device()
        
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        
        print(f"GPU Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        
        if required_gb and free_gb < required_gb:
            return False, f"Insufficient GPU memory: need {required_gb:.1f}GB, have {free_gb:.1f}GB"
        
        return True, f"GPU ready: {free_gb:.1f}GB available"
        
    except Exception as e:
        return False, f"GPU check failed: {e}"

def estimate_gpu_memory_needed(shape, dtype=np.uint8):
    """Estimate GPU memory needed for processing"""
    # Original + binary + distance (float32) + labeled (int32) + temp arrays
    bytes_per_voxel = 1 + 1 + 4 + 4 + 4  # Conservative estimate
    total_voxels = np.prod(shape)
    required_gb = (total_voxels * bytes_per_voxel) / (1024**3)
    return required_gb

def read_raw(filename, shape):
    """Reads a .raw file and returns a numpy array with the given shape."""
    print(f"Reading .raw file: {filename} with shape {shape}")
    data = np.fromfile(filename, dtype=np.uint8)
    expected_size = np.prod(shape)
    if data.size != expected_size:
        raise ValueError(f"Data size {data.size} does not match expected shape {shape} (requires {expected_size} elements).")
    image = data.reshape(shape)
    return image

def binarize_image(image, pore_value):
    """Binarizes the image based on the defined pore value."""
    unique_values = np.unique(image)
    print(f"Unique values in the image before binarization: {unique_values}")
    
    if pore_value not in unique_values:
        print(f"Warning: Specified pore value {pore_value} not found in the image. Proceeding with binarization based on presence of pore_value.")
    
    binary_image = (image == pore_value).astype(np.uint8)
    
    if not np.any(binary_image):
        print("Warning: No pores detected after binarization. Consider checking the pore_value or the input image.")
    
    return binary_image

def separate_pores_gpu(binary_image, use_gpu=True):
    """
    GPU-accelerated pore separation using CuPy.
    Falls back to CPU if GPU processing fails.
    
    Parameters:
        binary_image: 3D binary image where pores are 1 and grains are 0
        use_gpu: Whether to attempt GPU processing
    
    Returns:
        pore_sizes: List of pore sizes (in voxels)
        num_pores: Number of pores detected
        processing_info: Dict with processing details
    """
    processing_info = {"method": "unknown", "gpu_used": False, "timing": {}}
    
    if use_gpu and GPU_AVAILABLE:
        try:
            print("🚀 Starting GPU-accelerated pore separation...")
            start_time = time.time()
            
            # Transfer to GPU
            print("   Transferring data to GPU...")
            transfer_start = time.time()
            gpu_binary = cp.asarray(binary_image)
            processing_info["timing"]["gpu_transfer"] = time.time() - transfer_start
            
            # GPU Distance Transform (main bottleneck)
            print("   Performing GPU Euclidean Distance Transform...")
            distance_start = time.time()
            gpu_distance = gpu_ndi.distance_transform_edt(gpu_binary)
            processing_info["timing"]["gpu_distance"] = time.time() - distance_start
            
            # Transfer distance back to CPU for remaining operations
            print("   Transferring distance field back to CPU...")
            transfer_back_start = time.time()
            distance = cp.asnumpy(gpu_distance)
            processing_info["timing"]["gpu_transfer_back"] = time.time() - transfer_back_start
            
            # Clean up GPU memory
            del gpu_binary, gpu_distance
            cp.get_default_memory_pool().free_all_blocks()
            
            processing_info["method"] = "GPU distance + CPU watershed"
            processing_info["gpu_used"] = True
            
        except Exception as e:
            print(f"❌ GPU processing failed: {e}")
            print("   Falling back to CPU processing...")
            use_gpu = False
    
    if not use_gpu or not GPU_AVAILABLE:
        print("🖥️  Using CPU processing...")
        start_time = time.time()
        
        # CPU Distance Transform
        print("   Performing CPU Euclidean Distance Transform...")
        distance_start = time.time()
        distance = ndi.distance_transform_edt(binary_image)
        processing_info["timing"]["cpu_distance"] = time.time() - distance_start
        
        processing_info["method"] = "CPU only"
        processing_info["gpu_used"] = False
    
    # Remaining operations on CPU (these are relatively fast)
    print("   Identifying local maxima for watershed markers...")
    markers_start = time.time()
    local_maxi = morphology.local_maxima(distance)
    markers, num_features = ndi.label(local_maxi)
    processing_info["timing"]["markers"] = time.time() - markers_start
    print(f"   Number of initial markers found: {num_features}")
    
    if num_features == 0:
        print("   No markers found for watershed segmentation.")
        processing_info["timing"]["total"] = time.time() - start_time
        return [], 0, processing_info
    
    print("   Applying watershed segmentation to separate pores...")
    watershed_start = time.time()
    labeled_pores = segmentation.watershed(-distance, markers, mask=binary_image)
    processing_info["timing"]["watershed"] = time.time() - watershed_start
    
    num_pores = labeled_pores.max()
    print(f"   Number of pores detected after watershed: {num_pores}")
    
    if num_pores == 0:
        processing_info["timing"]["total"] = time.time() - start_time
        return [], 0, processing_info
    
    print("   Calculating pore sizes...")
    sizing_start = time.time()
    pore_sizes = ndi.sum(binary_image, labeled_pores, index=range(1, num_pores + 1))
    processing_info["timing"]["sizing"] = time.time() - sizing_start
    
    processing_info["timing"]["total"] = time.time() - start_time
    
    print(f"✅ Pore separation completed using {processing_info['method']}")
    print(f"   Total processing time: {processing_info['timing']['total']:.2f} seconds")
    
    return pore_sizes.tolist(), num_pores, processing_info

def filter_pore_sizes(pore_sizes, min_pore_size_voxels):
    """Filters out pore sizes smaller than the specified minimum size."""
    print(f"Filtering out pores smaller than {min_pore_size_voxels} voxels...")
    pore_sizes = np.array(pore_sizes)
    large_pores_mask = pore_sizes >= min_pore_size_voxels
    num_filtered = np.sum(~large_pores_mask)
    filtered_pore_sizes = pore_sizes[large_pores_mask]
    print(f"Number of pores after filtering: {filtered_pore_sizes.size}")
    print(f"Number of pores filtered out: {num_filtered}")
    return filtered_pore_sizes.tolist(), num_filtered

def plot_pore_size_distribution(pore_sizes, voxel_size=1.0, bins=50, output_path=None, log_scale=False):
    """Plots the pore size distribution."""
    print("Preparing data for pore size distribution plot...")
    pore_volumes = np.array(pore_sizes) * (voxel_size ** 3)
    pore_diameters = (6 * pore_volumes / np.pi) ** (1/3)
    
    mean_diameter = np.mean(pore_diameters)
    median_diameter = np.median(pore_diameters)
    print(f"Mean Pore Diameter: {mean_diameter:.6f} microns")
    print(f"Median Pore Diameter: {median_diameter:.6f} microns")
    
    plt.figure(figsize=(10, 6))
    plt.hist(pore_diameters, bins=bins, edgecolor='black', alpha=0.7, color='lightcoral')
    plt.axvline(mean_diameter, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_diameter:.3f} µm')
    plt.axvline(median_diameter, color='darkred', linestyle='dashed', linewidth=2, label=f'Median: {median_diameter:.3f} µm')
    plt.xlabel('Pore Diameter (µm)')
    plt.ylabel('Number of Pores')
    plt.title('Pore Size Distribution (GPU-Accelerated)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Pore Diameter (µm) [Log Scale]')
        plt.ylabel('Number of Pores [Log Scale]')
        plt.title('Pore Size Distribution (Log-Log Scale, GPU-Accelerated)')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pore size distribution plot saved to {output_path}")
    else:
        plt.show()
    plt.close()
    
    return mean_diameter, median_diameter

def save_pore_sizes(pore_sizes, output_file):
    """Saves the pore sizes to a text file."""
    print(f"Saving pore sizes to {output_file}...")
    np.savetxt(output_file, pore_sizes, fmt='%d')
    print("Pore sizes saved successfully.")

def save_statistics(mean, median, output_file):
    """Saves the mean and median pore sizes to a text file."""
    print(f"Saving statistics to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f"Mean Pore Diameter: {mean:.6f} microns\n")
        f.write(f"Median Pore Diameter: {median:.6f} microns\n")
    print("Statistics saved successfully.")

def save_performance_report(processing_info, output_file):
    """Save detailed performance report."""
    print(f"Saving performance report to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("=== GPU ACCELERATION PERFORMANCE REPORT ===\n\n")
        f.write(f"Processing Method: {processing_info['method']}\n")
        f.write(f"GPU Used: {processing_info['gpu_used']}\n")
        f.write(f"Total Time: {processing_info['timing']['total']:.3f} seconds\n\n")
        
        f.write("Detailed Timing Breakdown:\n")
        for stage, duration in processing_info['timing'].items():
            if stage != 'total':
                percentage = (duration / processing_info['timing']['total']) * 100
                f.write(f"  {stage}: {duration:.3f}s ({percentage:.1f}%)\n")
        
        if processing_info['gpu_used']:
            gpu_time = processing_info['timing'].get('gpu_distance', 0)
            transfer_time = (processing_info['timing'].get('gpu_transfer', 0) + 
                           processing_info['timing'].get('gpu_transfer_back', 0))
            f.write(f"\nGPU Efficiency:\n")
            f.write(f"  GPU Computation: {gpu_time:.3f}s\n")
            f.write(f"  Transfer Overhead: {transfer_time:.3f}s\n")
            f.write(f"  GPU Speedup Ratio: {gpu_time / (gpu_time + transfer_time):.2f}\n")
    
    print("Performance report saved successfully.")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='GPU-accelerated pore size distribution analysis.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input .raw file.')
    parser.add_argument('--shape', type=int, nargs=3, required=True, metavar=('DEPTH', 'HEIGHT', 'WIDTH'), help='Dimensions of the 3D image.')
    parser.add_argument('--voxel_size', type=float, default=1.0, help='Size of a voxel in microns (default: 1.0).')
    parser.add_argument('--bins', type=int, default=50, help='Number of bins for the histogram (default: 50).')
    parser.add_argument('--output_plot', type=str, default=None, help='Path to save the pore size distribution plot.')
    parser.add_argument('--output_sizes', type=str, default=None, help='Path to save the pore sizes as a text file.')
    parser.add_argument('--output_stats', type=str, default=None, help='Path to save the statistical analysis as a text file.')
    parser.add_argument('--output_performance', type=str, default=None, help='Path to save the performance report.')
    parser.add_argument('--pore_value', type=int, default=1, help='The pixel value that represents pores in the binary image (default: 1).')
    parser.add_argument('--log_scale', action='store_true', help='Use logarithmic scale for the histogram plot.')
    parser.add_argument('--min_pore_size', type=float, default=0.0, help='Minimum pore diameter in microns to be considered (default: 0.0, no filtering).')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU processing even if GPU is available.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    input_file = args.input
    shape = tuple(args.shape)
    voxel_size = args.voxel_size
    bins = args.bins
    output_plot = args.output_plot
    output_sizes = args.output_sizes
    output_stats = args.output_stats
    output_performance = args.output_performance
    pore_value = args.pore_value
    log_scale = args.log_scale
    min_pore_size = args.min_pore_size
    force_cpu = args.force_cpu
    
    if not os.path.isfile(input_file):
        print(f"Error: The input file {input_file} does not exist.")
        sys.exit(1)
    
    try:
        print("=== GPU-ACCELERATED PORE SIZE DISTRIBUTION ANALYSIS ===")
        print(f"Dataset shape: {shape}")
        print(f"Estimated data size: {np.prod(shape) / 1e6:.1f}M voxels")
        
        # Check GPU availability and memory
        use_gpu = not force_cpu
        if use_gpu and GPU_AVAILABLE:
            required_memory = estimate_gpu_memory_needed(shape)
            gpu_ok, gpu_msg = check_gpu_memory(required_memory)
            print(f"GPU Status: {gpu_msg}")
            if not gpu_ok:
                print("Falling back to CPU processing...")
                use_gpu = False
        elif force_cpu:
            print("CPU processing forced by user")
            use_gpu = False
        
        # Step 1: Read the .raw file
        image = read_raw(input_file, shape)
        
        # Step 2: Binarize the image based on pore_value
        binary_image = binarize_image(image, pore_value)
        
        # Step 3: GPU-accelerated pore separation
        pore_sizes, num_pores, processing_info = separate_pores_gpu(binary_image, use_gpu)
        
        print(f"\nTotal number of pores detected: {num_pores}")
        
        if num_pores == 0:
            print("No pores detected. Exiting the program.")
            sys.exit(0)
        
        # Step 4: Filter out small pores if min_pore_size is specified
        if min_pore_size > 0.0:
            min_pore_size_voxels = min_pore_size / voxel_size
            min_pore_size_voxels = max(1, int(np.ceil(min_pore_size_voxels)))
            pore_sizes, num_filtered = filter_pore_sizes(pore_sizes, min_pore_size_voxels)
            num_pores = len(pore_sizes)
            print(f"Number of pores after filtering: {num_pores}")
            print(f"Number of pores filtered out: {num_filtered}")
            
            if num_pores == 0:
                print("All pores were filtered out based on the minimum pore size threshold. Exiting the program.")
                sys.exit(0)
        else:
            print("No minimum pore size filtering applied.")
        
        # Step 5: Plot pore size distribution and calculate statistics
        mean_diameter, median_diameter = plot_pore_size_distribution(
            pore_sizes, voxel_size=voxel_size, bins=bins, output_path=output_plot, log_scale=log_scale
        )
        
        # Step 6: Save results if requested
        if output_sizes:
            save_pore_sizes(pore_sizes, output_sizes)
        if output_stats:
            save_statistics(mean_diameter, median_diameter, output_stats)
        if output_performance:
            save_performance_report(processing_info, output_performance)
        
        # Print performance summary
        print(f"\n🚀 PERFORMANCE SUMMARY:")
        print(f"   Method: {processing_info['method']}")
        print(f"   Total time: {processing_info['timing']['total']:.2f} seconds")
        if processing_info['gpu_used']:
            gpu_time = processing_info['timing'].get('gpu_distance', 0)
            cpu_equiv_time = processing_info['timing']['total'] + gpu_time * 9  # Rough estimate
            print(f"   Estimated CPU equivalent: ~{cpu_equiv_time:.1f} seconds")
            print(f"   Estimated speedup: ~{cpu_equiv_time / processing_info['timing']['total']:.1f}x")
        
        print("\n✅ GPU-accelerated processing completed successfully!")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()