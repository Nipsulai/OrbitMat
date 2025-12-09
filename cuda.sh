#!/bin/bash
set -e
# Load CP2K environment
source "$(pwd)/cp2k/tools/toolchain/install/setup"

# !!! Set the #threads you want
export OMP_NUM_THREADS=8

# Input file
INPUT="setup/tests/cp2k/large.inp"

# Run CP2K
./cp2k/exe/local_cuda/cp2k.ssmp -i "$INPUT" -o out.out