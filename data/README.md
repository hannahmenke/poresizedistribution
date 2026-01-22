# Data

This directory contains input data files for pore size distribution analysis.

All files in this directory are gitignored and will not be committed to the repository.

## File Format

Input files should be raw binary 3D images:
- Format: `.raw` (uint8 binary)
- Structure: 3D array with shape (depth, height, width)
- Values: Typically 0 for pores, 255 for solid material

## Example Usage

```bash
python src/PoreSizeDistPar.py \
    --input data/your_sample.raw \
    --shape 400 400 400 \
    --voxel_size 5.0 \
    --pore_value 0 \
    --output_plot outputs/results.png
```
