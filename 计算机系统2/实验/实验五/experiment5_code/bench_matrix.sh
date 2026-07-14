#!/bin/bash
cd ~/experiment5
echo "=== Matrix Multiplication Benchmark ==="
echo "Size    Naive(s)        Optimized(s)    Speedup"
echo "---------------------------------------------------"
for size in 100 500 1000 1500 2000 2500 3000; do
    n=$(./naive $size | awk '{print $2}')
    o=$(./optimized $size | awk '{print $2}')
    sp=$(echo "scale=3; $n / $o" | bc -l)
    printf '%-7d %-15s %-15s %s\n' $size $n $o $sp
done
echo "=== Done ==="
