# Implementation Guide

## Repository Structure

```
├── README.md
├── environment.yml
├── requirements.txt
├── setup_env.sh
├── data/                        # Input data files (gitignored)
│   └── README.md
├── docs/
│   ├── IMPLEMENTATIONS.md       # This file
│   └── QUICK_START.md
├── src/
│   ├── PoreSizeDist.py          # Original reference
│   ├── PoreSizeDistPar.py       # Recommended
│   ├── PoreSizeDistGPU.py       # GPU-accelerated
│   ├── PoreSizeDistGraph.py     # Most accurate (slower)
│   ├── PoreSizeDistOptimized.py # Adaptive processing
│   └── pore_render.py           # 3D mesh utilities
├── examples/
│   ├── benchmark.py
│   └── benchmark_real_data.py
└── outputs/                     # Generated outputs (gitignored)
    └── README.md
```

## Which Implementation to Use

**PoreSizeDistPar.py** - Default choice
```bash
python src/PoreSizeDistPar.py --input data/sample.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0
```

**PoreSizeDistGPU.py** - For GPU acceleration
```bash
python src/PoreSizeDistGPU.py --input data/sample.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0
```

**PoreSizeDistGraph.py** - For higher accuracy (slower)
```bash
python src/PoreSizeDistGraph.py --input data/sample.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0 --max_workers 4
```

**PoreSizeDistOptimized.py** - For automatic optimization
```bash
python src/PoreSizeDistOptimized.py --input data/sample.raw --shape 400 400 400 --voxel_size 5.0 --pore_value 0
```

## Utilities

**pore_render.py** - Exports 3D meshes (PLY/OBJ) for visualization. Integrated via `--render_output` parameter.

## Technical Notes

### Algorithm Details
Processing pipeline: Read raw file → Binarize → Distance transform → Watershed segmentation → Calculate volumes → Generate statistics

### Memory Usage
Typical 400³ dataset: ~1GB peak memory during processing

### Implementation Differences
- **PoreSizeDistPar.py**: Standard watershed segmentation
- **PoreSizeDistGPU.py**: GPU-accelerated distance transform with CPU fallback
- **PoreSizeDistGraph.py**: Connectivity-based analysis (slower, finds ~14% more pores)
- **PoreSizeDistOptimized.py**: Automatic selection based on data size