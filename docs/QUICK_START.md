# Quick Start Guide

## Setup

```bash
# Using conda
conda env create -f environment.yml
conda activate poresize-analysis

# Or using pip
pip install numpy scipy scikit-image matplotlib
```

## Basic Usage

```bash
python src/PoreSizeDistPar.py \
    --input data/your_data.raw \
    --shape 400 400 400 \
    --voxel_size 5.0 \
    --bins 100 \
    --pore_value 0 \
    --min_pore_size 20 \
    --output_plot outputs/results.png \
    --output_stats outputs/stats.txt
```

## Which Implementation

| Use Case | Implementation |
|----------|----------------|
| Most cases | `src/PoreSizeDistPar.py` |
| NVIDIA GPU | `src/PoreSizeDistGPU.py` |
| Higher accuracy | `src/PoreSizeDistGraph.py` |
| Auto-optimize | `src/PoreSizeDistOptimized.py` |

## Common Parameters

```bash
--input data.raw              # Your 3D image file
--shape 400 400 400           # Dimensions (depth height width)
--voxel_size 5.0              # Physical size in microns
--bins 100                    # Histogram bins for plot
--pore_value 0                # Pixel value representing pores
--min_pore_size 20            # Minimum pore diameter in microns
--output_plot results.png     # Save plot (optional)
--output_stats results.txt    # Save statistics (optional)
--render_output pores.ply     # Save 3D mesh of separated pores (optional)
--render_max_pores 200        # Limit pores in render (0 = all)
--render_min_size 50          # Min pore diameter for render (microns)
--render_max_size 500         # Max pore diameter for render (microns)
--subvolume_start 0 0 0       # Optional subvolume start (z y x)
--subvolume_size 200 400 400  # Optional subvolume size (dz dy dx)
--log_scale                   # Use log scale for plot (optional)
```

## Benchmarking

```bash
cd examples/
python benchmark.py                 # Small synthetic test
python benchmark_real_data.py       # Real dataset test
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Memory error | Use `PoreSizeDistOptimized.py` or reduce dataset size |
| GPU not found | Install CuPy or use `--force_cpu` flag |
| Slow performance | Ensure only one instance running |
| Wrong pore count | Verify `--pore_value` parameter |

## Documentation

- **README.md** - Full documentation
- **IMPLEMENTATIONS.md** - Technical details
- **QUICK_START.md** - This file
