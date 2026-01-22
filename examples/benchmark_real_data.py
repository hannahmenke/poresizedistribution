#!/usr/bin/env python3
# Benchmark script for real Bentheimer sandstone dataset
import time
import subprocess
import sys
import os

def benchmark_real_data(script_name, extra_args=None):
    """Benchmark script on real Bentheimer dataset"""
    print(f"\n{'='*60}")
    print(f"Testing {script_name} on Bentheimer 400³ dataset")
    print(f"{'='*60}")
    
    base_cmd = [
        sys.executable, script_name,
        "--input", "Bentheimer400-5mum.raw",
        "--shape", "400", "400", "400",
        "--voxel_size", "5.0",
        "--bins", "100", 
        "--pore_value", "0",
        "--min_pore_size", "20",
        "--output_plot", f"{script_name}_bentheimer_plot.png",
        "--output_sizes", f"{script_name}_bentheimer_sizes.txt",
        "--output_stats", f"{script_name}_bentheimer_stats.txt"
    ]
    
    if extra_args:
        base_cmd.extend(extra_args)
    
    print(f"Command: {' '.join(base_cmd)}")
    print("\nRunning...")
    
    start_time = time.time()
    try:
        result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        end_time = time.time()
        
        if result.returncode != 0:
            print(f"❌ Error running {script_name}:")
            print(result.stderr)
            return None
        
        execution_time = end_time - start_time
        
        # Extract key metrics from output
        output_lines = result.stdout.split('\n')
        num_pores = None
        mean_diameter = None
        median_diameter = None
        
        for line in output_lines:
            if "Total number of pores detected:" in line:
                num_pores = int(line.split(':')[1].strip())
            elif "Mean Pore Diameter:" in line:
                mean_diameter = float(line.split(':')[1].split()[0])
            elif "Median Pore Diameter:" in line:
                median_diameter = float(line.split(':')[1].split()[0])
        
        print(f"✅ Completed in {execution_time:.1f} seconds")
        print(f"   Pores detected: {num_pores}")
        print(f"   Mean diameter: {mean_diameter:.3f} µm")
        print(f"   Median diameter: {median_diameter:.3f} µm")
        
        return {
            'time': execution_time,
            'num_pores': num_pores,
            'mean_diameter': mean_diameter,
            'median_diameter': median_diameter,
            'success': True
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout running {script_name} (>30 minutes)")
        return None
    except Exception as e:
        print(f"❌ Exception running {script_name}: {e}")
        return None

def main():
    # Check if Bentheimer dataset exists
    if not os.path.isfile("Bentheimer400-5mum.raw"):
        print("❌ Bentheimer400-5mum.raw not found!")
        print("Place your Bentheimer dataset in the current directory and re-run.")
        return

    print("🧪 REAL-WORLD PERFORMANCE BENCHMARK")
    print("Dataset: Bentheimer sandstone 400³ voxels @ 5µm resolution")
    print("Size: 64M voxels, ~61MB file")
    print("Expected: Large number of complex pores")

    results = {}

    # Test each version
    versions = [
        ("PoreSizeDist.py", "Original Sequential", []),
        ("PoreSizeDistPar.py", "Fixed Parallel (Recommended)", []),
        ("PoreSizeDistGPU.py", "GPU Accelerated", []),
        ("PoreSizeDistGraph.py", "Graph-Based (Most Accurate)", ["--max_workers", "4"]),
        ("PoreSizeDistOptimized.py", "Adaptive Optimized", [])
    ]
    
    for script, name, extra_args in versions:
        if os.path.isfile(script):
            print(f"\n⏳ Testing {name}...")
            result = benchmark_real_data(script, extra_args)
            results[name] = result
        else:
            print(f"\n❌ {script} not found, skipping...")
            results[name] = None
    
    # Summary
    print(f"\n{'='*80}")
    print("🏆 REAL-WORLD PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if not successful_results:
        print("❌ No successful runs!")
        return
    
    # Find fastest
    fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
    
    print(f"{'Version':<25} {'Time (s)':<10} {'Speedup':<10} {'Pores':<8} {'Mean (µm)':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        if result is None:
            print(f"{name:<25} {'FAILED':<10} {'-':<10} {'-':<8} {'-':<10}")
        else:
            speedup = fastest[1]['time'] / result['time']
            print(f"{name:<25} {result['time']:<10.1f} {speedup:<10.2f}x {result['num_pores']:<8} {result['mean_diameter']:<10.3f}")
    
    print(f"\n🥇 Fastest: {fastest[0]} ({fastest[1]['time']:.1f} seconds)")
    
    # Check if results are consistent
    pore_counts = [r['num_pores'] for r in successful_results.values() if r['num_pores'] is not None]
    if pore_counts:
        min_pores, max_pores = min(pore_counts), max(pore_counts)
        pore_variation = (max_pores - min_pores) / min_pores * 100
        print(f"📊 Pore count variation: {pore_variation:.1f}% ({min_pores} to {max_pores})")
        
        if pore_variation < 5:
            print("✅ Results are consistent across implementations!")
        elif pore_variation < 15:
            print("⚠️  Some variation in results - may indicate algorithm differences")
        else:
            print("❌ Significant variation - check implementations for bugs")

if __name__ == "__main__":
    main()