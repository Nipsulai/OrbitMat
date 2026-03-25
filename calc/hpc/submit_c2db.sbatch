#!/bin/bash
# ── SLURM settings ────────────────────────────────────────────────────────────
#SBATCH --job-name=cp2k_c2db
#SBATCH --output=/groups/tensorlab/argatoff/OrbitMat/calc/hpc/logs/c2db_%A_%a.out
#SBATCH --error=/groups/tensorlab/argatoff/OrbitMat/calc/hpc/logs/c2db_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=2               # parallel jobs per node  (= config "parallel")
#SBATCH --cpus-per-task=2        # threads per job         (= config "threads")
#SBATCH --time=00:10:00
#SBATCH --array=0-1              # number of array tasks; set to ceil(N / CHUNK_SIZE) - 1

# ── Edit these ────────────────────────────────────────────────────────────────
REPO_DIR="/groups/tensorlab/argatoff/OrbitMat"
CONFIG="$REPO_DIR/calc/config_nm_tzvp_c2db.json"
SCRATCH="/resnick/scratch/nargatoff"
CHUNK_SIZE=347    # entries per array task; set to ceil(total / num_array_tasks)

# ── CP2K environment ──────────────────────────────────────────────────────────
module purge
module load openmpi/4.1.5-gcc-13.2.0-24q3ap2 \
            gcc/13.2.0-gcc-13.2.0-w55nxkl \
            gmp/6.2.1-gcc-13.2.0-lcnhyse \
            gsl/2.7.1-gcc-13.2.0-6hld2nf

source /groups/tensorlab/mnouman/miniconda3/etc/profile.d/conda.sh
conda activate OrbitMat

ulimit -s unlimited

export LD_LIBRARY_PATH="/home/mnouman/cp2k/build/bin:$LD_LIBRARY_PATH"
export CP2K_BIN="/home/mnouman/cp2k/build/bin/cp2k.psmp"

# ── Chunk math ────────────────────────────────────────────────────────────────
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
OUT_DIR="$SCRATCH/cp2k_out/c2db/tzvp_${SLURM_ARRAY_TASK_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: entries [${START_IDX}, ${END_IDX}) → ${OUT_DIR}"

cd "$REPO_DIR"
mkdir -p calc/hpc/logs "$OUT_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
python calc/hpc/run_slurm.py "$CONFIG" \
    --start "$START_IDX" \
    --end   "$END_IDX" \
    --out   "$OUT_DIR"