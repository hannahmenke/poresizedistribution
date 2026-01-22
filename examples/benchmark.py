#!/usr/bin/env python3
import time
import numpy as np
import subprocess
import sys
import os

def create_test_data(shape=(100, 100, 100), filename="test_data.raw"):
    """Create a small test dataset for benchmarking"""
    print(f"Creating test data with shape {shape}...")
    
    # Create a 3D volume with some pores
    np.random.seed(42)  # For reproducible results
    data = np.random.choice([0, 255], size=shape, p=[0.3, 0.7]).astype(np.uint8)
    
    # Add some structured pores (spheres)
    center = np.array(shape) // 2
    for i in range(5):
        sphere_center = center + np.random.randint(-20, 21, 3)
        radius = np.random.randint(3, 8)
        
        # Create sphere
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        mask = (z - sphere_center[0])**2 + (y - sphere_center[1])**2 + (x - sphere_center[2])**2 <= radius**2
        data[mask] = 0  # Pore value
    
    # Save to file
    data.tofile(filename)
    print(f"Test data saved to {filename}")
    return filename

def benchmark_script(script_name, input_file, shape, runs=3):
    """Benchmark a script multiple times and return average execution time"""
    times = []
    
    for i in range(runs):
        print(f"Running {script_name} - attempt {i+1}/{runs}...")
        
        cmd = [
            sys.executable, script_name,
            "--input", input_file,
            "--shape", str(shape[0]), str(shape[1]), str(shape[2]),
            "--voxel_size", "1.0",
            "--bins", "20",
            "--pore_value", "0",
            "--min_pore_size", "5"
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            if result.returncode != 0:
                print(f"Error running {script_name}:")
                print(result.stderr)
                return None
            
            execution_time = end_time - start_time
            times.append(execution_time)
            print(f"Execution time: {execution_time:.2f} seconds")
            
        except subprocess.TimeoutExpired:
            print(f"Timeout running {script_name}")
            return None
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"Average time for {script_name}: {avg_time:.2f} ± {std_time:.2f} seconds")
    
    return avg_time, std_time


def main():
    # Test parameters
    test_shape = (100, 100, 100)  # Small test for quick verification
    
    # Create test data
    test_file = create_test_data(test_shape)
    
    try:
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK - ALL IMPLEMENTATIONS")
        print("="*60)

        # Test original version
        print("\nTesting original version (PoreSizeDist.py)...")
        original_time = benchmark_script("../src/PoreSizeDist.py", test_file, test_shape)

        # Test fixed parallel version
        print("\nTesting fixed parallel version (PoreSizeDistPar.py)...")
        parallel_time = benchmark_script("../src/PoreSizeDistPar.py", test_file, test_shape)

        # Test GPU version (will use CPU fallback if no GPU)
        print("\nTesting GPU version (PoreSizeDistGPU.py)...")
        gpu_time = benchmark_script("../src/PoreSizeDistGPU.py", test_file, test_shape)

        # Test optimized version
        print("\nTesting optimized version (PoreSizeDistOptimized.py)...")
        optimized_time = benchmark_script("../src/PoreSizeDistOptimized.py", test_file, test_shape)

        # Compare results
        if original_time and parallel_time and gpu_time and optimized_time:
            print("\n" + "="*50)
            print("RESULTS COMPARISON")
            print("="*50)
            print(f"Original version:     {original_time[0]:.2f} ± {original_time[1]:.2f} seconds")
            print(f"Parallel version:     {parallel_time[0]:.2f} ± {parallel_time[1]:.2f} seconds")
            print(f"GPU version:          {gpu_time[0]:.2f} ± {gpu_time[1]:.2f} seconds")
            print(f"Optimized version:    {optimized_time[0]:.2f} ± {optimized_time[1]:.2f} seconds")

            speedup_parallel = original_time[0] / parallel_time[0]
            speedup_gpu = original_time[0] / gpu_time[0]
            speedup_optimized = original_time[0] / optimized_time[0]

            print(f"\nSpeedups vs Original:")
            print(f"Parallel:  {speedup_parallel:.2f}x {'FASTER' if speedup_parallel > 1 else 'SLOWER'}")
            print(f"GPU:       {speedup_gpu:.2f}x {'FASTER' if speedup_gpu > 1 else 'SLOWER'}")
            print(f"Optimized: {speedup_optimized:.2f}x {'FASTER' if speedup_optimized > 1 else 'SLOWER'}")

            best_time = min(original_time[0], parallel_time[0], gpu_time[0], optimized_time[0])
            if best_time == optimized_time[0]:
                print(f"\n🏆 BEST: Optimized version - {speedup_optimized:.2f}x faster than original")
            elif best_time == parallel_time[0]:
                print(f"\n🏆 BEST: Parallel version - {speedup_parallel:.2f}x faster than original")
            elif best_time == gpu_time[0]:
                print(f"\n🏆 BEST: GPU version - {speedup_gpu:.2f}x faster than original")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nCleaned up test file: {test_file}")

if __name__ == "__main__":
    main()