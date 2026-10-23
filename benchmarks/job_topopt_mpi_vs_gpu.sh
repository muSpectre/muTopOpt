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
# What is measured: the wall-clock time of each complete simulate.py
# invocation, in whole seconds (bash $SECONDS), i.e. including process and
# MPI start-up, module import, setup, FFT planning and any rocFFT kernel
# compilation, not only the optimization; no output file is written. The
# job log records the muGrid build (printed by _env.sh and by simulate.py),
# grid, iteration count, precision and the date. simulate.py's default
# optimizer ('auto') is used, i.e. trust-region Newton-CG if the installed
# NuMPI provides it, else L-BFGS.
#
# Note: `filtered_random` init is generated on the global grid and sliced per
# rank, so all three runs start from the same density; their inner CG
# iteration counts can still differ slightly because floating-point results
# depend on the backend (FFT library) and rank count (reduction order) — fine for a wall-clock comparison, not a
# convergence-trajectory comparison. Set SIZE/ITERS/RANKS/PRECISION below to
# taste; PRECISION=single runs the whole FFT-accelerated solve in float32 (the
# host-side outer optimizer stays double regardless).

set -euo pipefail
MUGRID_BUILD="${MUGRID_BUILD:-build_mpi}"
TOOLCHAIN="${TOOLCHAIN:-mi300a}"
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

SIZE="${SIZE:-128}"                              # cubic grid: SIZE^3
ITERS="${ITERS:-20}"                             # capped L-BFGS outer iterations
RANKS="${RANKS:-${SLURM_NTASKS:-92}}"            # CPU MPI ranks for the MPI run
GPU_RANKS="${GPU_RANKS:-${SLURM_GPUS_ON_NODE:-4}}"   # GPUs for the multi-GPU run
PRECISION="${PRECISION:-double}"                 # field/solve precision: single|double
# One control file per device; everything else at simulate.py's defaults.
control() {  # control DEVICE FILE
    cat > "$2" <<JSON
{
  "grid": {"nb_grid_pts": [$SIZE, $SIZE, $SIZE]},
  "target": {"bulk_modulus": 0.1, "shear_modulus": 0.05},
  "initial_density": {"source": "filtered_random", "seed": 0},
  "optimizer": {"max_iterations": $ITERS},
  "compute": {"device": "$1", "precision": "$PRECISION"},
  "output": {"file": "topopt_$1.nc"}
}
JSON
}
control cpu topopt_cpu.json
control gpu topopt_gpu.json

echo "=== muTopOpt benchmark: MPI ($RANKS CPU ranks) vs 1x GPU vs ${GPU_RANKS}x GPU ==="
echo "grid=${SIZE}^3  iters=$ITERS  precision=$PRECISION  $(date)"
echo

# --- Full-node MPI (CPU) --------------------------------------------------- #
echo ">>> [1/3] MPI on $RANKS CPU ranks (device=cpu)"
t0=$SECONDS
mpirun -np "$RANKS" python3 simulate.py topopt_cpu.json
t_mpi=$(( SECONDS - t0 ))
echo ">>> MPI wall time: ${t_mpi}s"
echo

# --- Single GPU ------------------------------------------------------------ #
echo ">>> [2/3] Single GPU (device=gpu)"
t0=$SECONDS
python3 simulate.py topopt_gpu.json
t_gpu=$(( SECONDS - t0 ))
echo ">>> GPU wall time: ${t_gpu}s"
echo

# --- Multi-GPU via MPI ----------------------------------------------------- #
# One rank per GPU.
echo ">>> [3/3] ${GPU_RANKS} GPUs via MPI (device=gpu)"
t0=$SECONDS
mpirun -np "$GPU_RANKS" python3 simulate.py topopt_gpu.json
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
