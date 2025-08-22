#!/usr/bin/env python3
# Phase 2: Chunk-based processing implementation
# This version processes data in chunks to reduce memory usage and enable better parallelization

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
    """
    Reads a .raw file and returns a numpy array with the given shape.
    """
    print(f"Reading .raw file: {filename} with shape {shape}")
    data = np.fromfile(filename, dtype=np.uint8)
    expected_size = np.prod(shape)
    if data.size != expected_size:
        raise ValueError(f"Data size {data.size} does not match expected shape {shape} (requires {expected_size} elements).")
    image = data.reshape(shape)
    return image

def binarize_image(image, pore_value):
    """
    Binarizes the image based on the defined pore value.
    """
    unique_values = np.unique(image)
    print(f"Unique values in the image before binarization: {unique_values}")
    
    if pore_value not in unique_values:
        print(f"Warning: Specified pore value {pore_value} not found in the image. Proceeding with binarization based on presence of pore_value.")
    
    binary_image = (image == pore_value).astype(np.uint8)
    
    if not np.any(binary_image):
        print("Warning: No pores detected after binarization. Consider checking the pore_value or the input image.")
    
    return binary_image

def process_chunk(chunk_data):
    """
    Process a single chunk of the binary image.
    Returns ONLY pore sizes for this chunk (no plotting).
    """
    chunk, chunk_id, overlap = chunk_data
    
    # Skip empty chunks
    if not np.any(chunk):
        return chunk_id, []
    
    try:
        # Distance transform for this chunk
        distance = ndi.distance_transform_edt(chunk)
        
        # Find local maxima
        local_maxi = morphology.local_maxima(distance)
        markers, num_features = ndi.label(local_maxi)
        
        if num_features == 0:
            return chunk_id, []
        
        # Watershed segmentation
        labeled_pores = segmentation.watershed(-distance, markers, mask=chunk)
        
        # Calculate pore sizes only
        num_pores = labeled_pores.max()
        if num_pores > 0:
            pore_sizes = ndi.sum(chunk, labeled_pores, index=range(1, num_pores + 1))
            return chunk_id, pore_sizes.tolist()
        else:
            return chunk_id, []
            
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {e}")
        return chunk_id, []

def create_chunks(binary_image, chunk_size=(64, 64, 64), overlap=8):
    """
    Create overlapping chunks from the binary image.
    
    Parameters:
        binary_image: 3D binary image
        chunk_size: Size of each chunk (z, y, x)
        overlap: Overlap between chunks to handle boundary pores
    
    Returns:
        List of (chunk, chunk_id, overlap) tuples
    """
    print(f"Creating chunks with size {chunk_size} and overlap {overlap}...")
    
    depth, height, width = binary_image.shape
    chunk_d, chunk_h, chunk_w = chunk_size
    
    chunks = []
    chunk_id = 0
    
    for z in range(0, depth, chunk_d - overlap):
        for y in range(0, height, chunk_h - overlap):
            for x in range(0, width, chunk_w - overlap):
                # Calculate actual chunk boundaries
                z_end = min(z + chunk_d, depth)
                y_end = min(y + chunk_h, height)
                x_end = min(x + chunk_w, width)
                
                # Extract chunk
                chunk = binary_image[z:z_end, y:y_end, x:x_end]
                
                # Only process non-empty chunks
                if np.any(chunk):
                    chunk_info = (z, y, x, z_end, y_end, x_end)
                    chunks.append((chunk, chunk_id, chunk_info))
                
                chunk_id += 1
    
    print(f"Created {len(chunks)} non-empty chunks for processing")
    return chunks

def merge_chunk_results(chunk_results, min_pore_size_voxels=1):
    """
    Merge results from all chunks and remove duplicates.
    
    Parameters:
        chunk_results: List of (chunk_id, pore_sizes) tuples
        min_pore_size_voxels: Minimum pore size to keep
    
    Returns:
        merged_pore_sizes: List of merged pore sizes
    """
    print("Merging chunk results...")
    
    all_pore_sizes = []
    total_pores_before_filter = 0
    
    for chunk_id, pore_sizes in chunk_results:
        if len(pore_sizes) > 0:
            total_pores_before_filter += len(pore_sizes)
            # Convert to numpy array for filtering
            pore_sizes_array = np.array(pore_sizes)
            # Filter by minimum size
            valid_sizes = pore_sizes_array[pore_sizes_array >= min_pore_size_voxels]
            all_pore_sizes.extend(valid_sizes.tolist())
    
    print(f"Total pores found across all chunks: {total_pores_before_filter}")
    print(f"Pores after size filtering: {len(all_pore_sizes)}")
    
    return all_pore_sizes

def separate_pores_chunked(binary_image, chunk_size=(64, 64, 64), overlap=8, max_workers=None):
    """
    Separates pores using chunked parallel processing.
    
    Parameters:
        binary_image: 3D binary image where pores are 1 and grains are 0
        chunk_size: Size of chunks for processing
        overlap: Overlap between chunks
        max_workers: Maximum number of worker processes
    
    Returns:
        pore_sizes: List of pore sizes (in voxels)
        num_pores: Number of pores detected
    """
    print("Starting chunked pore separation...")
    
    # Create chunks
    chunks = create_chunks(binary_image, chunk_size, overlap)
    
    if not chunks:
        print("No chunks to process!")
        return [], 0
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(len(chunks), mp.cpu_count())
    
    print(f"Processing {len(chunks)} chunks with {max_workers} workers...")
    
    # Process chunks in parallel
    chunk_results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(process_chunk, chunk_data): chunk_data[1] 
                          for chunk_data in chunks}
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                result = future.result()
                chunk_results.append(result)
                completed += 1
                if completed % 10 == 0 or completed == len(chunks):
                    print(f"Processed {completed}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Chunk {chunk_id} generated an exception: {e}")
                chunk_results.append((chunk_id, []))
    
    # Merge results
    pore_sizes = merge_chunk_results(chunk_results, min_pore_size_voxels=1)
    num_pores = len(pore_sizes)
    
    print(f"Chunked processing complete. Total pores detected: {num_pores}")
    return pore_sizes, num_pores

def filter_pore_sizes(pore_sizes, min_pore_size_voxels):
    """
    Filters out pore sizes smaller than the specified minimum size.
    """
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
    Plots the pore size distribution.
    """
    print("Preparing data for pore size distribution plot...")
    pore_volumes = np.array(pore_sizes) * (voxel_size ** 3)
    pore_diameters = (6 * pore_volumes / np.pi) ** (1/3)
    
    print("Calculating mean and median pore diameters...")
    mean_diameter = np.mean(pore_diameters)
    median_diameter = np.median(pore_diameters)
    print(f"Mean Pore Diameter: {mean_diameter:.6f} microns")
    print(f"Median Pore Diameter: {median_diameter:.6f} microns")
    
    print("Plotting pore size distribution...")
    plt.figure(figsize=(10, 6))
    plt.hist(pore_diameters, bins=bins, edgecolor='black', alpha=0.7)
    plt.axvline(mean_diameter, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_diameter:.6f} µm')
    plt.axvline(median_diameter, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_diameter:.6f} µm')
    plt.xlabel('Pore Diameter (µm)')
    plt.ylabel('Number of Pores')
    plt.title('Pore Size Distribution (Chunked Processing)')
    plt.legend()
    plt.grid(True)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Pore Diameter (µm) [Log Scale]')
        plt.ylabel('Number of Pores [Log Scale]')
        plt.title('Pore Size Distribution (Log-Log Scale, Chunked Processing)')
    
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

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Process a 3D binary .raw image using chunked parallel processing.')
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
    parser.add_argument('--chunk_size', type=int, nargs=3, default=[64, 64, 64], metavar=('CZ', 'CY', 'CX'), help='Chunk size for processing (default: 64 64 64).')
    parser.add_argument('--overlap', type=int, default=8, help='Overlap between chunks (default: 8).')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker processes (default: auto).')
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
    chunk_size = tuple(args.chunk_size)
    overlap = args.overlap
    max_workers = args.max_workers
    
    if not os.path.isfile(input_file):
        print(f"Error: The input file {input_file} does not exist.")
        sys.exit(1)
    
    try:
        print("=== CHUNKED PORE SIZE DISTRIBUTION ANALYSIS ===")
        print(f"Chunk size: {chunk_size}, Overlap: {overlap}, Max workers: {max_workers}")
        
        # Step 1: Read the .raw file
        image = read_raw(input_file, shape)
        
        # Step 2: Binarize the image based on pore_value
        binary_image = binarize_image(image, pore_value)
        
        # Step 3: Separate pores using chunked processing
        pore_sizes, num_pores = separate_pores_chunked(
            binary_image, chunk_size=chunk_size, overlap=overlap, max_workers=max_workers
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