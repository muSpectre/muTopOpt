#!/bin/bash
#SBATCH --job-name=muTopOpt-mpi-vs-gpu
#SBATCH --partition=mi300a
#SBATCH --nodes=1
#SBATCH --ntasks=92
#SBATCH --cpus-per-task=1
#SBATCH --gpus=4
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --account=bw17d009
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Simple topology-optimization benchmark: full-node MPI (CPU) vs 1 GPU vs 4 GPUs.
#
# Runs the SAME fixed-size, fixed-iteration-count 3D stress-matching design three
# times — once distributed over all CPU cores with mpirun, once on one MI300A GPU,
# and once across 4 GPUs via MPI — and reports the wall-clock time of each. Every
# config does the same amount of per-iteration work (identical grid, 6 load cases,
# `--bfgs-maxiter` outer steps run to a capped count rather than to convergence), so the
# ratios are a rough throughput comparison of the FFT-accelerated forward/adjoint
# solve across the backends.
#
# Note: `filtered_random` init draws per-rank noise, so the CPU (multi-rank) and
# GPU (single-rank) runs do not start from a bit-identical density and their inner
# CG iteration counts can differ slightly — fine for a wall-clock comparison, not
# a convergence-trajectory comparison. Set SIZE/ITERS/RANKS/PRECISION below to
# taste; PRECISION=single runs the whole FFT-accelerated solve in float32 (the
# host L-BFGS optimizer stays double regardless).

set -euo pipefail
MUGRID_BUILD="${MUGRID_BUILD:-build_mpi}"
TOOLCHAIN="${TOOLCHAIN:-mi300a}"
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

SIZE="${SIZE:-128}"                              # cubic grid: SIZE^3
ITERS="${ITERS:-20}"                             # capped L-BFGS outer iterations
RANKS="${RANKS:-${SLURM_NTASKS:-92}}"            # CPU MPI ranks for the MPI run
GPU_RANKS="${GPU_RANKS:-${SLURM_GPUS_ON_NODE:-4}}"   # GPUs for the multi-GPU run
PRECISION="${PRECISION:-double}"                 # field/solve precision: single|double
COMMON=(-n "$SIZE" "$SIZE" "$SIZE" --target-K 0.1 --target-G 0.05 \
        --bfgs-maxiter "$ITERS" --seed 0 --init filtered_random \
        --precision "$PRECISION")

echo "=== muTopOpt benchmark: MPI ($RANKS CPU ranks) vs 1x GPU vs ${GPU_RANKS}x GPU ==="
echo "grid=${SIZE}^3  iters=$ITERS  precision=$PRECISION  $(date)"
echo

# --- Full-node MPI (CPU) --------------------------------------------------- #
echo ">>> [1/3] MPI on $RANKS CPU ranks (device=cpu)"
t0=$SECONDS
mpirun -np "$RANKS" python3 simulate.py "${COMMON[@]}" --device cpu
t_mpi=$(( SECONDS - t0 ))
echo ">>> MPI wall time: ${t_mpi}s"
echo

# --- Single GPU ------------------------------------------------------------ #
echo ">>> [2/3] Single GPU (device=gpu)"
t0=$SECONDS
python3 simulate.py "${COMMON[@]}" --device gpu
t_gpu=$(( SECONDS - t0 ))
echo ">>> GPU wall time: ${t_gpu}s"
echo

# --- Multi-GPU via MPI ----------------------------------------------------- #
# One rank per GPU; UCX_TLS=^rocm_ipc (set in _env.sh) is required here.
echo ">>> [3/3] ${GPU_RANKS} GPUs via MPI (device=gpu)"
t0=$SECONDS
mpirun -np "$GPU_RANKS" python3 simulate.py "${COMMON[@]}" --device gpu
t_gpu4=$(( SECONDS - t0 ))
echo ">>> ${GPU_RANKS}x GPU wall time: ${t_gpu4}s"
echo

# --- Summary --------------------------------------------------------------- #
echo "=== Summary (grid=${SIZE}^3, iters=$ITERS, precision=$PRECISION) ==="
printf "  MPI (%s CPU ranks): %ss\n" "$RANKS" "$t_mpi"
printf "  1x GPU           : %ss\n" "$t_gpu"
printf "  %sx GPU           : %ss\n" "$GPU_RANKS" "$t_gpu4"
if (( t_gpu > 0 )); then
    printf "  speedup (MPI / 1 GPU) : %.2fx\n" "$(echo "scale=4; $t_mpi / $t_gpu" | bc)"
fi
if (( t_gpu4 > 0 )); then
    printf "  speedup (1 GPU / %s GPU): %.2fx\n" "$GPU_RANKS" "$(echo "scale=4; $t_gpu / $t_gpu4" | bc)"
fi
