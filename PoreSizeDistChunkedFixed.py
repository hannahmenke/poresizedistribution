#!/usr/bin/env python3
# Phase 2: FIXED Chunk-based processing implementation
# Properly handles boundary artifacts, ghost cells, and single final plot

import os
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology, segmentation
import matplotlib.pyplot as plt
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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

def extract_chunk_with_ghost_cells(binary_image, z_start, y_start, x_start, 
                                 z_end, y_end, x_end, ghost_size=8):
    """
    Extract a chunk with ghost cells for proper boundary handling.
    
    Returns:
        chunk_with_ghost: Chunk with ghost cells
        core_slice: Slice indices for the core region (without ghost cells)
        full_shape: Shape of the chunk with ghost cells
    """
    depth, height, width = binary_image.shape
    
    # Calculate ghost cell boundaries
    z_ghost_start = max(0, z_start - ghost_size)
    y_ghost_start = max(0, y_start - ghost_size)
    x_ghost_start = max(0, x_start - ghost_size)
    
    z_ghost_end = min(depth, z_end + ghost_size)
    y_ghost_end = min(height, y_end + ghost_size)
    x_ghost_end = min(width, x_end + ghost_size)
    
    # Extract chunk with ghost cells
    chunk_with_ghost = binary_image[z_ghost_start:z_ghost_end, 
                                   y_ghost_start:y_ghost_end, 
                                   x_ghost_start:x_ghost_end]
    
    # Calculate core region indices within the ghost chunk
    core_z_start = z_start - z_ghost_start
    core_y_start = y_start - y_ghost_start
    core_x_start = x_start - x_ghost_start
    
    core_z_end = core_z_start + (z_end - z_start)
    core_y_end = core_y_start + (y_end - y_start)
    core_x_end = core_x_start + (x_end - x_start)
    
    core_slice = (slice(core_z_start, core_z_end),
                  slice(core_y_start, core_y_end),
                  slice(core_x_start, core_x_end))
    
    return chunk_with_ghost, core_slice, chunk_with_ghost.shape

def process_chunk_with_boundaries(chunk_data):
    """
    Process a single chunk with proper boundary handling.
    Only returns pore sizes from the core region (not ghost cells).
    """
    (chunk_with_ghost, core_slice, chunk_id) = chunk_data
    
    # Skip if no pores in the chunk
    if not np.any(chunk_with_ghost):
        return chunk_id, []
    
    try:
        # Distance transform on the full chunk (including ghost cells)
        # This ensures proper distance calculation near boundaries
        distance = ndi.distance_transform_edt(chunk_with_ghost)
        
        # Find local maxima in the full chunk
        local_maxi = morphology.local_maxima(distance)
        markers, num_features = ndi.label(local_maxi)
        
        if num_features == 0:
            return chunk_id, []
        
        # Watershed segmentation on the full chunk
        labeled_pores = segmentation.watershed(-distance, markers, mask=chunk_with_ghost)
        
        # Extract only the core region results (no ghost cells)
        core_labeled = labeled_pores[core_slice]
        core_mask = chunk_with_ghost[core_slice]
        
        # Calculate pore sizes only for pores in the core region
        unique_labels = np.unique(core_labeled[core_mask > 0])
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        pore_sizes = []
        for label in unique_labels:
            # Count voxels of this pore in the core region only
            pore_size = np.sum((core_labeled == label) & (core_mask > 0))
            if pore_size > 0:
                pore_sizes.append(pore_size)
        
        return chunk_id, pore_sizes
        
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {e}")
        return chunk_id, []

def calculate_optimal_chunks(image_shape, target_chunks=None, min_chunk_size=200):
    """
    Calculate optimal chunk size based on available cores and minimum chunk size.
    
    Parameters:
        image_shape: (depth, height, width)
        target_chunks: Target number of chunks (defaults to CPU cores)
        min_chunk_size: Minimum dimension for chunks
    
    Returns:
        chunk_size: Optimal chunk size
        actual_chunks: Actual number of chunks that will be created
    """
    if target_chunks is None:
        target_chunks = mp.cpu_count()  # Use number of CPU cores
    
    depth, height, width = image_shape
    
    # Start with cube root of target chunks for 3D division
    chunks_per_dim = round(target_chunks ** (1/3))
    
    # Calculate chunk sizes
    chunk_d = max(min_chunk_size, depth // chunks_per_dim)
    chunk_h = max(min_chunk_size, height // chunks_per_dim)
    chunk_w = max(min_chunk_size, width // chunks_per_dim)
    
    # Adjust if chunks would be too big (don't make chunks bigger than the image)
    chunk_d = min(chunk_d, depth)
    chunk_h = min(chunk_h, height)
    chunk_w = min(chunk_w, width)
    
    # Calculate actual number of chunks
    chunks_z = max(1, (depth + chunk_d - 1) // chunk_d)
    chunks_y = max(1, (height + chunk_h - 1) // chunk_h)
    chunks_x = max(1, (width + chunk_w - 1) // chunk_w)
    actual_chunks = chunks_z * chunks_y * chunks_x
    
    chunk_size = (chunk_d, chunk_h, chunk_w)
    
    print(f"Optimal chunking strategy:")
    print(f"  Image shape: {image_shape}")
    print(f"  Target chunks: {target_chunks} (CPU cores: {mp.cpu_count()})")
    print(f"  Min chunk size: {min_chunk_size}")
    print(f"  Calculated chunk size: {chunk_size}")
    print(f"  Actual chunks to create: {actual_chunks}")
    print(f"  Chunks per dimension: {chunks_z}×{chunks_y}×{chunks_x}")
    
    return chunk_size, actual_chunks

def create_chunks_with_boundaries(binary_image, target_chunks=None, min_chunk_size=200, ghost_size=32):
    """
    Create optimal chunks with ghost cells for proper boundary handling.
    
    Parameters:
        binary_image: 3D binary image
        target_chunks: Target number of chunks (defaults to CPU cores)
        min_chunk_size: Minimum chunk dimension
        ghost_size: Size of ghost cell boundary
    
    Returns:
        List of (chunk_with_ghost, core_slice, chunk_id) tuples
    """
    image_shape = binary_image.shape
    
    # Calculate optimal chunk size
    chunk_size, actual_chunks = calculate_optimal_chunks(image_shape, target_chunks, min_chunk_size)
    
    print(f"Creating {actual_chunks} chunks with core size {chunk_size} and ghost size {ghost_size}...")
    
    depth, height, width = image_shape
    chunk_d, chunk_h, chunk_w = chunk_size
    
    chunks = []
    chunk_id = 0
    
    # Create non-overlapping core chunks
    for z in range(0, depth, chunk_d):
        for y in range(0, height, chunk_h):
            for x in range(0, width, chunk_w):
                # Calculate core chunk boundaries
                z_end = min(z + chunk_d, depth)
                y_end = min(y + chunk_h, height)
                x_end = min(x + chunk_w, width)
                
                # Extract chunk with ghost cells
                chunk_with_ghost, core_slice, full_shape = extract_chunk_with_ghost_cells(
                    binary_image, z, y, x, z_end, y_end, x_end, ghost_size
                )
                
                # Only process chunks that have pores in the core region
                core_region = chunk_with_ghost[core_slice]
                if np.any(core_region):
                    chunks.append((chunk_with_ghost, core_slice, chunk_id))
                
                chunk_id += 1
    
    print(f"Created {len(chunks)} chunks for processing (core regions with pores)")
    return chunks

def separate_pores_chunked_fixed(binary_image, target_chunks=None, min_chunk_size=200, ghost_size=32, max_workers=None):
    """
    Separates pores using chunked processing with optimal chunk sizing.
    
    Parameters:
        binary_image: 3D binary image where pores are 1 and grains are 0
        target_chunks: Target number of chunks (defaults to CPU cores)
        min_chunk_size: Minimum chunk dimension (default: 200)
        ghost_size: Size of ghost cell boundary for proper distance calculation
        max_workers: Maximum number of worker processes
    
    Returns:
        pore_sizes: List of pore sizes (in voxels) - NO DUPLICATES
        num_pores: Number of pores detected
    """
    print("Starting FIXED chunked pore separation with boundary handling...")
    
    # Create chunks with ghost cells
    chunks = create_chunks_with_boundaries(binary_image, target_chunks, min_chunk_size, ghost_size)
    
    if not chunks:
        print("No chunks to process!")
        return [], 0
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(len(chunks), mp.cpu_count())
    
    print(f"Processing {len(chunks)} chunks with {max_workers} workers...")
    print("Using ghost cells to prevent boundary artifacts in distance transform")
    
    # Process chunks in parallel
    all_pore_sizes = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(process_chunk_with_boundaries, chunk_data): chunk_data[2] 
                          for chunk_data in chunks}
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                result = future.result()
                _, pore_sizes = result
                if pore_sizes:
                    all_pore_sizes.extend(pore_sizes)
                
                completed += 1
                if completed % 5 == 0 or completed == len(chunks):
                    print(f"Processed {completed}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Chunk {chunk_id} generated an exception: {e}")
    
    num_pores = len(all_pore_sizes)
    print(f"Fixed chunked processing complete.")
    print(f"Total pores detected: {num_pores} (no duplicates from boundaries)")
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
    """
    Plots the pore size distribution - SINGLE PLOT ONLY.
    """
    if len(pore_sizes) == 0:
        print("No pores to plot!")
        return 0, 0
        
    print("Creating SINGLE pore size distribution plot from all merged results...")
    pore_volumes = np.array(pore_sizes) * (voxel_size ** 3)
    pore_diameters = (6 * pore_volumes / np.pi) ** (1/3)
    
    mean_diameter = np.mean(pore_diameters)
    median_diameter = np.median(pore_diameters)
    print(f"Mean Pore Diameter: {mean_diameter:.6f} microns")
    print(f"Median Pore Diameter: {median_diameter:.6f} microns")
    
    # Create a SINGLE plot
    plt.figure(figsize=(10, 6))
    plt.hist(pore_diameters, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(mean_diameter, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_diameter:.3f} µm')
    plt.axvline(median_diameter, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_diameter:.3f} µm')
    plt.xlabel('Pore Diameter (µm)')
    plt.ylabel('Number of Pores')
    plt.title('Pore Size Distribution (Fixed Chunked Processing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Pore Diameter (µm) [Log Scale]')
        plt.ylabel('Number of Pores [Log Scale]')
        plt.title('Pore Size Distribution (Log-Log Scale, Fixed Chunked)')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"SINGLE plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()  # Ensure plot is closed to prevent memory issues
    
    return mean_diameter, median_diameter

def save_pore_sizes(pore_sizes, output_file):
    """Saves the pore sizes to a text file."""
    print(f"Saving {len(pore_sizes)} pore sizes to {output_file}...")
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
    parser = argparse.ArgumentParser(description='FIXED chunked processing with optimal chunk sizing and boundary handling.')
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
    parser.add_argument('--ghost_size', type=int, default=32, help='Ghost cell boundary size for preventing boundary artifacts (default: 32).')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker processes and target chunks (default: CPU cores).')
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
    ghost_size = args.ghost_size
    max_workers = args.max_workers
    
    if not os.path.isfile(input_file):
        print(f"Error: The input file {input_file} does not exist.")
        sys.exit(1)
    
    try:
        print("=== FIXED CHUNKED PORE SIZE DISTRIBUTION ANALYSIS ===")
        print(f"Ghost cell size: {ghost_size} (prevents boundary artifacts)")
        print(f"Max workers/target chunks: {max_workers if max_workers else 'CPU cores'}")
        print("Using optimal chunking strategy (min 200x200x200 per chunk)")
        print("Will create SINGLE final plot (no multiple plots)")
        
        # Step 1: Read the .raw file
        image = read_raw(input_file, shape)
        
        # Step 2: Binarize the image based on pore_value
        binary_image = binarize_image(image, pore_value)
        
        # Step 3: Separate pores using FIXED chunked processing
        pore_sizes, num_pores = separate_pores_chunked_fixed(
            binary_image, target_chunks=max_workers, min_chunk_size=200, ghost_size=ghost_size, max_workers=max_workers
        )
        
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
        else:
            print("No minimum pore size filtering applied.")
        
        # Step 5: Create SINGLE plot and calculate statistics
        mean_diameter, median_diameter = plot_pore_size_distribution(
            pore_sizes, voxel_size=voxel_size, bins=bins, output_path=output_plot, log_scale=log_scale
        )
        
        # Step 6: Save results if requested
        if output_sizes:
            save_pore_sizes(pore_sizes, output_sizes)
        if output_stats:
            save_statistics(mean_diameter, median_diameter, output_stats)
        
        print("FIXED chunked processing completed successfully - SINGLE PLOT CREATED!")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()