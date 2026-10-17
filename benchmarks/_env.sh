# Shared environment for the muTopOpt benchmark job scripts.
#
# Sourced by the job_*.sh scripts. Sets up the mi300a toolchain, puts the fresh
# build of muGrid (plus muTopOpt itself) on PYTHONPATH, and applies the UCX
# transport workaround needed for the multi-rank MPI runs. Does not set `set -e`
# — the caller owns that.

# --------------------------------------------------------------------------- #
# Repository layout, derived from this file's own location
# (<muSpectre>/muTopOpt/benchmarks/_env.sh) so any checkout works.
# --------------------------------------------------------------------------- #
MUTOPOPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MUGRID="$(dirname "$MUTOPOPT")/muGrid"

# Which muGrid build to benchmark, and the toolchain that built it. The two go
# together: build_nompi is the single-rank GPU build made with the mi300a_nompi
# venv. For a multi-rank run, build muGrid against the MPI toolchain and select
# it here, e.g. `MUGRID_BUILD=build_mpi TOOLCHAIN=mi300a`.
MUGRID_BUILD="${MUGRID_BUILD:-build_nompi}"
TOOLCHAIN="${TOOLCHAIN:-mi300a_nompi}"

if [ ! -d "$MUGRID/$MUGRID_BUILD" ]; then
    echo "_env.sh: no muGrid build at $MUGRID/$MUGRID_BUILD" >&2
    echo "         (build it, or select another one with MUGRID_BUILD=...)" >&2
    return 1
fi

# --------------------------------------------------------------------------- #
# Cluster module / toolchain environment (compilers, MPI, Python, ROCm, venv).
# --------------------------------------------------------------------------- #
source "/work/classic/fr_lp1029-IMTEK-Simulation/$TOOLCHAIN/env.sh"

# Footgun: a plain `import muGrid` picks up whatever copy is installed in the
# venv's site-packages, which need not be the build under test. Putting the
# build tree first on PYTHONPATH makes the fresh build win. muTopOpt is a plain
# source tree, so its repo root goes on the path too (simulate.py imports the
# local `muTopOpt` package).
export PYTHONPATH="$MUGRID/$MUGRID_BUILD/language_bindings/python:$MUGRID/language_bindings/python:$MUTOPOPT${PYTHONPATH:+:$PYTHONPATH}"

# ... and verify it did, because a benchmark of the wrong binary looks exactly
# like a benchmark of the right one.
python3 -c "
import sys, muGrid
ext = muGrid._muGrid.__file__
if not ext.startswith('$MUGRID/$MUGRID_BUILD/'):
    sys.exit(f'_env.sh: benchmarking the wrong muGrid: {ext}')
print(f'muGrid: {muGrid.version_string()}')
print(f'  package   {muGrid.__file__}')
print(f'  extension {ext}')
"

# rocFFT compiles its kernels at runtime (the `fft_rtc_*` names in a kernel
# trace). Without a persistent cache every process recompiles them from
# scratch, which costs minutes per (grid, precision) before the first solve --
# absorbed by the benchmark's warmup, so it does not distort ms/CG-iter, but it
# makes a sweep far slower than the work it measures.
export ROCFFT_RTC_CACHE_PATH="${ROCFFT_RTC_CACHE_PATH:-$HOME/.cache/rocfft_kernels.db}"

# UCX_TLS=^rocm_ipc: the ROCm IPC rendezvous transport triggers an rkey-size
# assertion failure between ranks; disable it so UCX falls back to
# rocm_copy / shared-memory transfers. Harmless for the pure-CPU MPI run.
export UCX_TLS="^rocm_ipc"

cd "$MUTOPOPT"
