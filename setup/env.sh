#!/bin/bash
set -e

# GPU architecture
# Allowed values are:
# 20X, K40, K80, P100, V100, A100, H100, A4
GPU_VER="A100"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Paths
CP2K_DIR="./cp2k"
ENV_NAME="OrbitMat"

# Absolute output path
SETUP_DIR="$(pwd)/setup"
LOGFILE="$SETUP_DIR/out.log"

mkdir -p "$SETUP_DIR"
echo -e "${CYAN}=== CP2K GPU INSTALL SCRIPT ===${NC}" | tee "$LOGFILE"

log() {
    echo -e "$1" | tee -a "$LOGFILE"
}

# Check CUDA
log "${CYAN} Checking CUDA...${NC}"
if ! command -v nvcc &> /dev/null; then
    log "${RED}CUDA (nvcc) not found.${NC}"
    exit 1
fi

CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
log "${GREEN}CUDA version: $CUDA_VER${NC}"

# Update submodule
log "${CYAN}\nUpdating CP2K submodule...${NC}"
if [ ! -d "$CP2K_DIR" ]; then
    log "${RED}CP2K submodule not found at '$CP2K_DIR'.${NC}"
    exit 1
fi

git submodule update --init --recursive "$CP2K_DIR" 2>&1 | tee -a "$LOGFILE"

# Run toolchain
cd "$CP2K_DIR/tools/toolchain"

log "${CYAN}\n Running CP2K toolchain...${NC}"
./install_cp2k_toolchain.sh \
    --with-gcc=install \
    --with-libxc=install \
    --with-fftw=install \
    --with-libint=install \
    --with-libxsmm=install \
    --with-openblas=install \
    --enable-cuda \
    --gpu-ver="$GPU_VER" \
    2>&1 | tee -a "$LOGFILE"

log "${GREEN}Toolchain complete.${NC}"

# Return to CP2K root directory
cd ../..

# Prepare architecture files
mkdir -p arch
cp tools/toolchain/install/arch/* arch/
source tools/toolchain/install/setup

# Build CP2K (GPU)
log "${CYAN}\nBuilding CP2K (cuda)...${NC}"
make -j "$(nproc)" ARCH=local_cuda VERSION=ssmp \
    2>&1 | tee "$SETUP_DIR/build_gpu.log"

# Build CP2K (CPU)
log "${CYAN}\nBuilding CP2K (CPU)...${NC}"
make -j "$(nproc)" ARCH=local VERSION=ssmp \
    2>&1 | tee "$SETUP_DIR/build_cpu.log"

log "${GREEN}CP2K build complete.${NC}"

# Create Conda environment
log "${CYAN}\nCreating Conda environment '$ENV_NAME'...${NC}"
eval "$(conda shell.bash hook)"

if conda env list | grep -q "$ENV_NAME"; then
    log "${YELLOW}Environment already exists. Skipping.${NC}"
else
    conda create -n "$ENV_NAME" python=3.10 -y 2>&1 | tee -a "$LOGFILE"
fi

conda activate "$ENV_NAME"
pip install numpy pandas scipy matplotlib pymatgen ipykernel ase 2>&1 | tee -a "$LOGFILE"

log "${GREEN}\n=== Installation Complete ===${NC}"
log "Executables: $CP2K_DIR/exe/local_cuda/"
log "Conda env:  $ENV_NAME"
log "Logs saved in: $SETUP_DIR/"
