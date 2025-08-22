# python PoreSizeDistPar.py \
#     --input Bentheimer400-5mum.raw \
#     --shape 400 400 400 \
#     --voxel_size 5 \
#     --bins 100 \
#     --output_plot pore_distribution.png \
#     --output_sizes pore_sizes.txt \
#     --output_stats pore_stats.txt \
#     --pore_value 0 \
#     --min_pore_size 20 \
#     --log_scale


# Note: Voxel size should be in microns!

import os

import numpy as np
import scipy.ndimage as ndi
from skimage import morphology, segmentation
import matplotlib.pyplot as plt
import argparse
import sys

def read_raw(filename, shape):
    """
    Reads a .raw file and returns a numpy array with the given shape.
    
    Parameters:
        filename (str): Path to the .raw file.
        shape (tuple): Dimensions of the 3D image (depth, height, width).
    
    Returns:
        np.ndarray: 3D numpy array representing the binary image.
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
    
    Parameters:
        image (np.ndarray): Input image.
        pore_value (int): The value in the image that represents pores.
    
    Returns:
        np.ndarray: Binary image where pores are 1 and grains are 0.
    """
    unique_values = np.unique(image)
    print(f"Unique values in the image before binarization: {unique_values}")
    
    if pore_value not in unique_values:
        print(f"Warning: Specified pore value {pore_value} not found in the image. Proceeding with binarization based on presence of pore_value.")
    
    # Binarize: pores are where image == pore_value
    binary_image = (image == pore_value).astype(np.uint8)
    
    # Check if binary_image has any pores
    if not np.any(binary_image):
        print("Warning: No pores detected after binarization. Consider checking the pore_value or the input image.")
    
    return binary_image

def separate_pores_skimage(binary_image):
    """
    Separates pores using scikit-image's watershed segmentation.
    
    Parameters:
        binary_image (np.ndarray): 3D binary image where pores are 1 and grains are 0.
    
    Returns:
        labeled_pores (np.ndarray): Labeled image where each pore has a unique label.
        num_pores (int): Number of pores detected.
        pore_sizes (list): List of pore sizes (in voxels).
    """
    print("Performing Euclidean Distance Transform...")
    distance = ndi.distance_transform_edt(binary_image)
    
    print("Identifying local maxima for watershed markers...")
    local_maxi = morphology.local_maxima(distance)
    markers, num_features = ndi.label(local_maxi)
    print(f"Number of initial markers found: {num_features}")
    
    print("Applying watershed segmentation to separate pores...")
    labeled_pores = segmentation.watershed(-distance, markers, mask=binary_image)
    
    num_pores = labeled_pores.max()
    print(f"Number of pores detected after watershed: {num_pores}")
    
    print("Calculating pore sizes...")
    pore_sizes = ndi.sum(binary_image, labeled_pores, index=range(1, num_pores + 1))
    
    return labeled_pores, num_pores, pore_sizes

def filter_pore_sizes(pore_sizes, min_pore_size_voxels):
    """
    Filters out pore sizes smaller than the specified minimum size.
    
    Parameters:
        pore_sizes (list or np.ndarray): List of pore sizes in voxels.
        min_pore_size_voxels (int): Minimum pore size in voxels.
    
    Returns:
        filtered_pore_sizes (list): List of pore sizes after filtering.
        num_filtered (int): Number of pores filtered out.
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
    
    Parameters:
        pore_sizes (list or np.ndarray): List of pore sizes in voxels.
        voxel_size (float): Size of a voxel in microns.
        bins (int): Number of bins for the histogram.
        output_path (str): Path to save the plot. If None, displays the plot.
        log_scale (bool): If True, uses a logarithmic scale for the axes.
    
    Returns:
        mean_diameter (float): Mean pore diameter in microns.
        median_diameter (float): Median pore diameter in microns.
    """
    print("Preparing data for pore size distribution plot...")
    # Convert pore sizes to physical units (microns)
    pore_volumes = np.array(pore_sizes) * (voxel_size ** 3)
    
    # Compute equivalent spherical diameters in microns
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
    plt.title('Pore Size Distribution')
    plt.legend()
    plt.grid(True)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Pore Diameter (µm) [Log Scale]')
        plt.ylabel('Number of Pores [Log Scale]')
        plt.title('Pore Size Distribution (Log-Log Scale)')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pore size distribution plot saved to {output_path}")
    else:
        plt.show()
    plt.close()
    
    return mean_diameter, median_diameter

def save_pore_sizes(pore_sizes, output_file):
    """
    Saves the pore sizes to a text file.
    
    Parameters:
        pore_sizes (list or np.ndarray): List of pore sizes in voxels.
        output_file (str): Path to the output text file.
    """
    print(f"Saving pore sizes to {output_file}...")
    np.savetxt(output_file, pore_sizes, fmt='%d')
    print("Pore sizes saved successfully.")

def save_statistics(mean, median, output_file):
    """
    Saves the mean and median pore sizes to a text file.
    
    Parameters:
        mean (float): Mean pore diameter in microns.
        median (float): Median pore diameter in microns.
        output_file (str): Path to the output text file.
    """
    print(f"Saving statistics to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f"Mean Pore Diameter: {mean:.6f} microns\n")
        f.write(f"Median Pore Diameter: {median:.6f} microns\n")
    print("Statistics saved successfully.")

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process a 3D binary .raw image to separate pores and generate pore size distribution with statistical analysis.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input .raw file.')
    parser.add_argument('--shape', type=int, nargs=3, required=True, metavar=('DEPTH', 'HEIGHT', 'WIDTH'), help='Dimensions of the 3D image.')
    parser.add_argument('--voxel_size', type=float, default=1.0, help='Size of a voxel in microns (default: 1.0).')
    parser.add_argument('--bins', type=int, default=50, help='Number of bins for the histogram (default: 50).')
    parser.add_argument('--output_plot', type=str, default=None, help='Path to save the pore size distribution plot (e.g., "pore_distribution.png"). If not provided, the plot will be displayed.')
    parser.add_argument('--output_sizes', type=str, default=None, help='Path to save the pore sizes as a text file (e.g., "pore_sizes.txt").')
    parser.add_argument('--output_stats', type=str, default=None, help='Path to save the statistical analysis (mean and median) as a text file (e.g., "pore_stats.txt").')
    parser.add_argument('--pore_value', type=int, default=1, help='The pixel value that represents pores in the binary image (default: 1).')
    parser.add_argument('--log_scale', action='store_true', help='Use logarithmic scale for the histogram plot.')
    parser.add_argument('--min_pore_size', type=float, default=0.0, help='Minimum pore diameter in microns to be considered (default: 0.0, no filtering).')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    input_file = args.input
    shape = tuple(args.shape)  # (depth, height, width)
    voxel_size = args.voxel_size
    bins = args.bins
    output_plot = args.output_plot
    output_sizes = args.output_sizes
    output_stats = args.output_stats
    pore_value = args.pore_value
    log_scale = args.log_scale
    min_pore_size = args.min_pore_size
    
    if not os.path.isfile(input_file):
        print(f"Error: The input file {input_file} does not exist.")
        sys.exit(1)
    
    try:
        # Step 1: Read the .raw file
        image = read_raw(input_file, shape)
        
        # Step 2: Binarize the image based on pore_value
        binary_image = binarize_image(image, pore_value)
        
        # Step 3: Separate pores and label them using scikit-image
        labeled_pores, num_pores, pore_sizes = separate_pores_skimage(binary_image)
        
        print(f"Total number of pores detected: {num_pores}")
        
        if num_pores == 0:
            print("No pores detected. Exiting the program.")
            sys.exit(0)
        
        # Step 4: Filter out small pores if min_pore_size is specified
        if min_pore_size > 0.0:
            # Convert min_pore_size from microns to voxels
            min_pore_size_voxels = min_pore_size / voxel_size
            min_pore_size_voxels = max(1, int(np.ceil(min_pore_size_voxels)))  # Ensure at least 1 voxel
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
        
        # Step 6: Save pore sizes and statistics if requested
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