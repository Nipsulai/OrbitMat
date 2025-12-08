#!/bin/bash
set -e

# GPU architecture
GPU_VER="V100"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Paths
CP2K_DIR="./cp2k"
ENV_NAME="OrbitMat"
LOGFILE="./setup/out.log"

# Create setup directory
echo -e "${CYAN}=== CP2K GPU INSTALL SCRIPT ===${NC}" | tee "$LOGFILE"

# Logging function
log() {
    echo -e "$1" | tee -a "$LOGFILE"
}

# Check CUDA
log "${CYAN}Step 1: Checking CUDA...${NC}"
if ! command -v nvcc &> /dev/null; then
    log "${RED}CUDA (nvcc) not found.${NC}"
    exit 1
fi
CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
log "${GREEN}CUDA version: $CUDA_VER${NC}"

# Update CP2K submodule
log "${CYAN}\nStep 2: Updating CP2K submodule...${NC}"
if [ ! -d "$CP2K_DIR" ]; then
    log "${RED}CP2K submodule not found at '$CP2K_DIR'.${NC}"
    exit 1
fi
git submodule update --init --recursive "$CP2K_DIR" 2>&1 | tee -a "$LOGFILE"
cd "$CP2K_DIR"

# Run CP2K toolchain
log "${CYAN}\nStep 3: Running CP2K toolchain...${NC}"
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

# Prepare architecture files
mkdir -p arch
cp tools/toolchain/install/arch/* arch/
source tools/toolchain/install/setup

# Build CP2K (GPU)
log "${CYAN}\nStep 4: Building CP2K (GPU)...${NC}"
make -j "$(nproc)" ARCH=local_cuda VERSION=ssmp \
    2>&1 | tee "$SETUP_DIR/build_gpu.log"

# Build CP2K (CPU)
log "${CYAN}\nStep 5: Building CP2K (CPU)...${NC}"
make -j "$(nproc)" ARCH=local VERSION=ssmp \
    2>&1 | tee "$SETUP_DIR/build_cpu.log"
log "${GREEN}CP2K build complete.${NC}"

# Setup conda env
log "${CYAN}\nStep 6: Creating Conda environment '$ENV_NAME'...${NC}"
eval "$(conda shell.bash hook)"

if conda env list | grep -q "$ENV_NAME"; then
    log "${YELLOW}Environment already exists. Skipping.${NC}"
else
    conda create -n "$ENV_NAME" python=3.10 -y 2>&1 | tee -a "$LOGFILE"
fi

conda activate "$ENV_NAME"
pip install numpy pandas scipy matplotlib pymatgen 2>&1 | tee -a "$LOGFILE"

# Done
log "${GREEN}\n=== Installation Complete ===${NC}"
log "Executables: $CP2K_DIR/exe/local_cuda/"
log "Conda env:  $ENV_NAME"
log "Logs saved in: $SETUP_DIR/"