#!/usr/bin/env python3
# Phase 2: Optimized version with adaptive processing
# Uses chunking for large datasets, optimized algorithms for small ones

import os
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology, segmentation
import matplotlib.pyplot as plt
import argparse
import sys
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def get_memory_info():
    """Get available system memory in GB"""
    memory = psutil.virtual_memory()
    return memory.available / (1024**3)

def estimate_memory_usage(shape, dtype=np.uint8):
    """Estimate memory usage for processing this shape"""
    # Original image + binary image + distance transform + labeled image
    bytes_per_voxel = np.dtype(dtype).itemsize
    total_voxels = np.prod(shape)
    
    # Estimate: original + binary + distance (float64) + labeled (int32) + temp arrays
    estimated_gb = (total_voxels * (bytes_per_voxel + 1 + 8 + 4 + 8)) / (1024**3)
    return estimated_gb

def should_use_chunking(shape, available_memory_gb, safety_factor=0.5):
    """Determine if chunking should be used based on memory constraints"""
    estimated_usage = estimate_memory_usage(shape)
    threshold = available_memory_gb * safety_factor
    
    print(f"Estimated memory usage: {estimated_usage:.2f} GB")
    print(f"Available memory: {available_memory_gb:.2f} GB")
    print(f"Memory threshold: {threshold:.2f} GB")
    
    return estimated_usage > threshold

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

def separate_pores_optimized(binary_image):
    """
    Optimized pore separation using scipy with better memory management.
    """
    print("Performing optimized Euclidean Distance Transform...")
    # Use float32 instead of float64 to save memory
    distance = ndi.distance_transform_edt(binary_image).astype(np.float32)
    
    print("Identifying local maxima for watershed markers...")
    # Use more efficient local maxima detection
    local_maxi = morphology.local_maxima(distance)
    markers, num_features = ndi.label(local_maxi)
    print(f"Number of initial markers found: {num_features}")
    
    if num_features == 0:
        print("No markers found for watershed segmentation.")
        return [], 0
    
    print("Applying watershed segmentation to separate pores...")
    # Use more memory-efficient watershed
    labeled_pores = segmentation.watershed(-distance, markers, mask=binary_image, compactness=0.1)
    
    num_pores = labeled_pores.max()
    print(f"Number of pores detected after watershed: {num_pores}")
    
    if num_pores == 0:
        return [], 0
    
    print("Calculating pore sizes using optimized method...")
    # Use bincount for faster pore size calculation
    labeled_flat = labeled_pores.ravel()
    mask_flat = binary_image.ravel()
    
    # Only count voxels where the mask is True
    valid_labels = labeled_flat[mask_flat > 0]
    pore_sizes = np.bincount(valid_labels)[1:]  # Skip background (label 0)
    
    return pore_sizes.tolist(), num_pores

def process_chunk_optimized(chunk_data):
    """Optimized chunk processing"""
    chunk, chunk_id, chunk_info = chunk_data
    
    if not np.any(chunk):
        return chunk_id, []
    
    try:
        # Use optimized processing for chunks
        distance = ndi.distance_transform_edt(chunk).astype(np.float32)
        local_maxi = morphology.local_maxima(distance)
        markers, num_features = ndi.label(local_maxi)
        
        if num_features == 0:
            return chunk_id, []
        
        labeled_pores = segmentation.watershed(-distance, markers, mask=chunk, compactness=0.1)
        
        # Fast pore size calculation
        labeled_flat = labeled_pores.ravel()
        mask_flat = chunk.ravel()
        valid_labels = labeled_flat[mask_flat > 0]
        
        if len(valid_labels) > 0:
            pore_sizes = np.bincount(valid_labels)[1:]  # Skip background
            return chunk_id, pore_sizes.tolist()
        else:
            return chunk_id, []
            
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {e}")
        return chunk_id, []

def separate_pores_chunked_optimized(binary_image, chunk_size=(128, 128, 128), overlap=16, max_workers=None):
    """
    Chunked processing with optimizations
    """
    print("Starting optimized chunked pore separation...")
    
    depth, height, width = binary_image.shape
    chunk_d, chunk_h, chunk_w = chunk_size
    
    chunks = []
    chunk_id = 0
    
    print(f"Creating optimized chunks with size {chunk_size} and overlap {overlap}...")
    
    for z in range(0, depth, chunk_d - overlap):
        for y in range(0, height, chunk_h - overlap):
            for x in range(0, width, chunk_w - overlap):
                z_end = min(z + chunk_d, depth)
                y_end = min(y + chunk_h, height)
                x_end = min(x + chunk_w, width)
                
                chunk = binary_image[z:z_end, y:y_end, x:x_end]
                
                if np.any(chunk):
                    chunk_info = (z, y, x, z_end, y_end, x_end)
                    chunks.append((chunk, chunk_id, chunk_info))
                
                chunk_id += 1
    
    print(f"Created {len(chunks)} non-empty chunks for processing")
    
    if not chunks:
        return [], 0
    
    if max_workers is None:
        max_workers = min(len(chunks), mp.cpu_count())
    
    print(f"Processing chunks with {max_workers} workers...")
    
    all_pore_sizes = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk_optimized, chunk_data): chunk_data[1] 
                          for chunk_data in chunks}
        
        completed = 0
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                result = future.result()
                _, pore_sizes = result
                if pore_sizes:
                    all_pore_sizes.extend(pore_sizes)
                
                completed += 1
                if completed % 10 == 0 or completed == len(chunks):
                    print(f"Processed {completed}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Chunk {chunk_id} generated an exception: {e}")
    
    num_pores = len(all_pore_sizes)
    print(f"Optimized chunked processing complete. Total pores detected: {num_pores}")
    return all_pore_sizes, num_pores

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
    plt.hist(pore_diameters, bins=bins, edgecolor='black', alpha=0.7)
    plt.axvline(mean_diameter, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_diameter:.6f} µm')
    plt.axvline(median_diameter, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_diameter:.6f} µm')
    plt.xlabel('Pore Diameter (µm)')
    plt.ylabel('Number of Pores')
    plt.title('Pore Size Distribution (Optimized Processing)')
    plt.legend()
    plt.grid(True)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Pore Diameter (µm) [Log Scale]')
        plt.ylabel('Number of Pores [Log Scale]')
        plt.title('Pore Size Distribution (Log-Log Scale, Optimized)')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
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

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Process a 3D binary .raw image with adaptive optimization.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input .raw file.')
    parser.add_argument('--shape', type=int, nargs=3, required=True, metavar=('DEPTH', 'HEIGHT', 'WIDTH'), help='Dimensions of the 3D image.')
    parser.add_argument('--voxel_size', type=float, default=1.0, help='Size of a voxel in microns (default: 1.0).')
    parser.add_argument('--bins', type=int, default=50, help='Number of bins for the histogram (default: 50).')
    parser.add_argument('--output_plot', type=str, default=None, help='Path to save the pore size distribution plot.')
    parser.add_argument('--output_sizes', type=str, default=None, help='Path to save the pore sizes as a text file.')
    parser.add_argument('--output_stats', type=str, default=None, help='Path to save the statistical analysis as a text file.')
    parser.add_argument('--pore_value', type=int, default=1, help='The pixel value that represents pores in the binary image (default: 1).')
    parser.add_argument('--log_scale', action='store_true', help='Use logarithmic scale for the histogram plot.')
    parser.add_argument('--min_pore_size', type=float, default=0.0, help='Minimum pore diameter in microns to be considered (default: 0.0, no filtering).')
    parser.add_argument('--force_chunking', action='store_true', help='Force chunked processing regardless of memory constraints.')
    parser.add_argument('--chunk_size', type=int, nargs=3, default=[128, 128, 128], metavar=('CZ', 'CY', 'CX'), help='Chunk size for processing (default: 128 128 128).')
    parser.add_argument('--overlap', type=int, default=16, help='Overlap between chunks (default: 16).')
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
    pore_value = args.pore_value
    log_scale = args.log_scale
    min_pore_size = args.min_pore_size
    force_chunking = args.force_chunking
    chunk_size = tuple(args.chunk_size)
    overlap = args.overlap
    
    if not os.path.isfile(input_file):
        print(f"Error: The input file {input_file} does not exist.")
        sys.exit(1)
    
    try:
        print("=== OPTIMIZED PORE SIZE DISTRIBUTION ANALYSIS ===")
        
        # Check memory constraints
        available_memory = get_memory_info()
        use_chunking = force_chunking or should_use_chunking(shape, available_memory)
        
        if use_chunking:
            print(f"Using CHUNKED processing (chunk size: {chunk_size}, overlap: {overlap})")
        else:
            print("Using OPTIMIZED single-pass processing")
        
        # Step 1: Read the .raw file
        image = read_raw(input_file, shape)
        
        # Step 2: Binarize the image based on pore_value
        binary_image = binarize_image(image, pore_value)
        
        # Step 3: Choose processing method based on memory constraints
        if use_chunking:
            pore_sizes, num_pores = separate_pores_chunked_optimized(
                binary_image, chunk_size=chunk_size, overlap=overlap
            )
        else:
            pore_sizes, num_pores = separate_pores_optimized(binary_image)
        
        print(f"Total number of pores detected: {num_pores}")
        
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
        
        # Step 5: Plot pore size distribution and calculate statistics
        mean_diameter, median_diameter = plot_pore_size_distribution(
            pore_sizes, voxel_size=voxel_size, bins=bins, output_path=output_plot, log_scale=log_scale
        )
        
        # Step 6: Save results if requested
        if output_sizes:
            save_pore_sizes(pore_sizes, output_sizes)
        if output_stats:
            save_statistics(mean_diameter, median_diameter, output_stats)
        
        print("Processing completed successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()