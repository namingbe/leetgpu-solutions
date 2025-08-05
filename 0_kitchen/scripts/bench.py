#!/usr/bin/env python3
# kitchen/scripts/bench.py

import subprocess
import sys
import os
import torch
import numpy as np
from pathlib import Path
import importlib.util
import pandas as pd
import argparse

def load_module(filepath):
    spec = importlib.util.spec_from_file_location("baseline", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def benchmark_cuda(problem_dir, size):
    binary = problem_dir / "build" / problem_dir.name
    if not binary.exists():
        print(f"❌ CUDA binary not found: {binary}")
        return None
    
    result = subprocess.run([str(binary), str(size)], 
                          capture_output=True, text=True)
    # Parse output for timing
    for line in result.stdout.split('\n'):
        if 'Average time:' in line:
            return float(line.split()[2])
    return None

def benchmark_pytorch(problem_dir, size):
    baseline_file = problem_dir / "baseline_pytorch.py"
    if not baseline_file.exists():
        return None
    
    module = load_module(baseline_file)
    return module.benchmark(size)

def benchmark_tinygrad(problem_dir, size):
    baseline_file = problem_dir / "baseline_tinygrad.py"
    if not baseline_file.exists():
        return None
    
    module = load_module(baseline_file)
    return module.benchmark(size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='Problem directory name')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--csv', help='Output CSV file')
    args = parser.parse_args()
    
    problem_dir = Path(args.problem)
    
    results = {
        'CUDA': benchmark_cuda(problem_dir, args.size),
        'PyTorch': benchmark_pytorch(problem_dir, args.size),
        'Tinygrad': benchmark_tinygrad(problem_dir, args.size)
    }
    
    # Pretty print results
    print(f"\n📊 Benchmark Results for {args.problem} (size={args.size})")
    print("=" * 50)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        baseline = min(valid_results.values())
        for impl, time_ms in sorted(valid_results.items(), key=lambda x: x[1]):
            speedup = baseline / time_ms if time_ms > 0 else 0
            print(f"{impl:12} {time_ms:8.3f} ms  ({speedup:.2f}x)")
    
    if args.csv:
        df = pd.DataFrame([results])
        df['problem'] = args.problem
        df['size'] = args.size
        
        if Path(args.csv).exists():
            existing = pd.read_csv(args.csv)
            df = pd.concat([existing, df])
        
        df.to_csv(args.csv, index=False)

if __name__ == "__main__":
    main()

