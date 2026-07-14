#!/bin/bash
cd ~/experiment5
echo "=== Compiling ==="
gcc -O2 wsl_matmul.c -o matmul -Wall -lm
echo "=== Running benchmarks ==="
echo ""

for sz in 100 500 1000 1500 2000 2500 3000; do
    echo "--- Size = $sz ---"
    timeout 300 ./matmul $sz
    echo ""
done

echo "=== All done ==="
