#!/bin/bash

# Setup script for pore size distribution analysis environment

echo "Creating conda environment for pore size distribution analysis..."

# Create conda environment from yml file
conda env create -f environment.yml

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "conda activate poresize-analysis"
echo ""
echo "Then you can run the benchmarks and tests."