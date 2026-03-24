#!/bin/bash
# ── SLURM settings ────────────────────────────────────────────────────────────
#SBATCH --job-name=cp2k_orbitmat
#SBATCH --output=calc/hpc/logs/slurm_%A_%a.out
#SBATCH --error=calc/hpc/logs/slurm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=8               # parallel jobs per node  (= config "parallel")
#SBATCH --cpus-per-task=4        # threads per job         (= config "threads")
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --array=0-9              # number of array tasks; set to ceil(N / CHUNK_SIZE) - 1

# ── Edit these ────────────────────────────────────────────────────────────────
REPO_DIR="/home/mnouman/OrbitMat"
CONFIG="$REPO_DIR/calc/config_nm.json"
INPUT_JSON="/home/mnouman/data/bandgap2025/filtered/run_0324_1141/data.json"
SCRATCH="/resnick/groups/tensorlab/nargatoff"
CHUNK_SIZE=350    # entries per array task; set to ceil(total / num_array_tasks)

# ── CP2K environment ──────────────────────────────────────────────────────────
module purge
module load openmpi/4.1.5-gcc-13.2.0-24q3ap2 \
            gcc/13.2.0-gcc-13.2.0-w55nxkl \
            gmp/6.2.1-gcc-13.2.0-lcnhyse \
            gsl/2.7.1-gcc-13.2.0-6hld2nf

source /groups/tensorlab/mnouman/miniconda3/etc/profile.d/conda.sh
conda activate qmenv

ulimit -s unlimited

export CP2K_SETUP="/home/mnouman/cp2k/tools/toolchain/install/setup"
export CP2K_BIN="/home/mnouman/cp2k/build/bin/cp2k.ssmp"
export CP2K_DATA_DIR="/home/mnouman/cp2k/data"
export LD_LIBRARY_PATH="/home/mnouman/cp2k/build/bin:$LD_LIBRARY_PATH"

# ── Chunk math ────────────────────────────────────────────────────────────────
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
OUT_DIR="$SCRATCH/cp2k_out/tzvp_${SLURM_ARRAY_TASK_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: entries [${START_IDX}, ${END_IDX}) → ${OUT_DIR}"
mkdir -p calc/hpc/logs "$OUT_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$REPO_DIR"
python calc/hpc/run_slurm.py "$CONFIG" \
    --start "$START_IDX" \
    --end   "$END_IDX" \
    --out   "$OUT_DIR"
