# Pore Size Distribution Analysis

Tools for analyzing pore size distributions in 3D microstructural images. Multiple implementations available with different speed/accuracy trade-offs.

## Available Implementations

| Implementation | When to Use |
|----------------|-------------|
| **PoreSizeDistPar.py** | Default choice for most cases |
| **PoreSizeDistGPU.py** | When you have an NVIDIA GPU |
| **PoreSizeDistGraph.py** | When you need maximum accuracy (slower) |
| **PoreSizeDistOptimized.py** | For automatic optimization based on data size |
| **PoreSizeDist.py** | Original reference implementation |

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate poresize-analysis

# Or install manually
pip install numpy scipy scikit-image matplotlib
```

### 2. Basic Usage

```bash
# Recommended: Fixed parallel version (best performance/simplicity)
python src/PoreSizeDistPar.py \
    --input your_data.raw \
    --shape 400 400 400 \
    --voxel_size 5.0 \
    --bins 100 \
    --pore_value 0 \
    --min_pore_size 20 \
    --output_plot outputs/pore_distribution.png \
    --output_sizes outputs/pore_sizes.txt \
    --output_stats outputs/pore_stats.txt
```

### 3D Mesh Rendering (Optional)

```bash
# Export separated pores as a 3D mesh (PLY or OBJ)
python src/PoreSizeDistPar.py \
    --input your_data.raw \
    --shape 400 400 400 \
    --voxel_size 5.0 \
    --pore_value 0 \
    --render_output outputs/pores.ply \
    --render_max_pores 200
```

Optional render filters:
- `--render_min_size` / `--render_max_size` to cap pore diameters (microns)

### Subvolume Processing (Optional)

```bash
# Process only a subvolume (z y x start, dz dy dx size)
python src/PoreSizeDistPar.py \
    --input your_data.raw \
    --shape 400 400 400 \
    --subvolume_start 50 100 100 \
    --subvolume_size 100 200 200 \
    --pore_value 0
```

### View In ParaView

1. Open ParaView.
2. File → Open… → select the exported `.ply` or `.obj`.
3. Click Apply in the Properties panel.
4. Set Representation to `Surface` (or `Surface With Edges`).
5. If colors are missing, set Coloring to `RGB`.

If the mesh is heavy, use Filters → `Decimate` to reduce triangles.

## 📁 Implementation Details

### **PoreSizeDistPar.py** (Recommended)
Default choice for most users. Clean, reliable implementation.

```bash
python src/PoreSizeDistPar.py --input data.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0
```

### **PoreSizeDistGPU.py**
Uses GPU acceleration if available, otherwise falls back to CPU. Requires CuPy for GPU support.

```bash
python src/PoreSizeDistGPU.py --input data.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0
```

### **PoreSizeDistGraph.py**
Uses connectivity analysis for more accurate segmentation. Slower but finds more pores in complex networks.

```bash
python src/PoreSizeDistGraph.py --input data.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0 --max_workers 4
```

### **PoreSizeDistOptimized.py**
Automatically adapts processing strategy based on dataset size and available memory.

```bash
python src/PoreSizeDistOptimized.py --input data.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0
```

### **PoreSizeDist.py**
Original reference implementation. Use for comparison or educational purposes.

```bash
python src/PoreSizeDist.py --input data.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0
```

## Algorithm

The analysis pipeline:
1. Read .raw binary file
2. Binarize image (identify pore voxels)
3. Distance transform
4. Watershed segmentation
5. Calculate pore volumes
6. Generate size distribution statistics

## Benchmarking

```bash
cd examples/
python benchmark.py                    # Small synthetic test
python benchmark_real_data.py          # Real dataset test
```

## Advanced Usage

```bash
# High-resolution analysis
python src/PoreSizeDistPar.py \
    --input high_res_data.raw \
    --shape 1000 1000 1000 \
    --voxel_size 1.0 \
    --bins 200 \
    --pore_value 0 \
    --min_pore_size 5 \
    --log_scale \
    --output_plot outputs/high_res_distribution.png

# Graph-based analysis with custom settings
python src/PoreSizeDistGraph.py \
    --input data.raw \
    --shape 500 500 500 \
    --voxel_size 2.0 \
    --max_workers 8 \
    --output_report outputs/detailed_analysis.txt

# GPU with performance monitoring
python src/PoreSizeDistGPU.py --input data.raw --output_performance outputs/gpu_timing.txt

# Force CPU mode
python src/PoreSizeDistGPU.py --input data.raw --force_cpu
```

## Requirements

**Python packages:**
- Python 3.8+
- NumPy, SciPy, scikit-image, matplotlib

**Optional:**
- CuPy (for GPU acceleration)
- psutil (for memory monitoring)

Install with:
```bash
conda env create -f environment.yml
# or
pip install numpy scipy scikit-image matplotlib
```

## Troubleshooting

**Memory errors:** Use PoreSizeDistOptimized.py or reduce dataset size
**GPU not detected:** Install CuPy or use `--force_cpu` flag
**Slow performance:** Ensure only one instance is running
