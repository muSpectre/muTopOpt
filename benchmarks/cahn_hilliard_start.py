#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
A random Cahn-Hilliard (spinodal) start design for ``large_3d.py`` on the
``N³`` grid of the optimisation.

    python benchmarks/cahn_hilliard_start.py 128 CH128_s11.nc [--seed 11]

The procedure of the muFFTTO study's ``make_CH512_start.py``: the periodic
Cahn-Hilliard equation

    ∂c/∂t = Δ(c³ − c − ε² Δc)

from noise ``c = 0.05 N(0, 1)`` around the mean 0, with semi-implicit
spectral steps ``dt = 0.0625 ε²`` up to ``t = 250 ε²`` and the discrete
7-point Laplacian symbol. Its interface width is the optimisation's:
``ε = √2 η`` with ``η = 2h`` (``h = 1/N``), so that the interface
``c = tanh(x / (√2 ε))`` is the equilibrium profile
``ρ = (1 + tanh(x / (2η))) / 2`` of the phase-field regularization. The
structures therefore get finer with the grid. The design is
``ρ = (1 − c) / 2``, clipped to ``[0, 1]`` (no threshold, no blur).

Writes the design in the study's layout (variable ``phase_field_solution``
of shape ``(1, 1, N, N, N)``), which ``large_3d.py --start`` and the study's
renderers read, and a PNG of the mid slice next to it. Serial; on the GPU
(cupy) when available, else numpy.
"""
import argparse
import time

import numpy as np


def array_module(device):
    """cupy for ``device == "gpu"`` (or "auto" with a GPU), else numpy."""
    if device in ("gpu", "auto"):
        try:
            import cupy

            cupy.zeros(1)
            return cupy
        except Exception:
            if device == "gpu":
                raise
    return np


def cahn_hilliard(n, seed=11, eta_h=2.0, eps_eta=np.sqrt(2.0),
                  t_eps2=250.0, dt_eps2=0.0625, xp=np, log=print):
    """The concentration ``c`` (host array ``(n, n, n)``) after the
    Cahn-Hilliard evolution on the periodic unit cube from seeded noise.

    Parameters
    ----------
    n : int
        Grid points per axis.
    seed : int
        Seed of the noise (numpy's ``default_rng``).
    eta_h : float
        Interface width of the optimisation in grid spacings, ``η = eta_h h``.
    eps_eta : float
        ``ε / η``; ``√2`` gives the regularization's equilibrium profile.
    t_eps2, dt_eps2 : float
        End time and time step in units of ``ε²``.
    xp : module
        numpy or cupy.
    log : callable
        Progress lines (ten per run); ``None`` for silence.
    """
    h = 1.0 / n
    eps = eps_eta * eta_h * h
    dt = dt_eps2 * eps ** 2
    steps = int(round(t_eps2 / dt_eps2))
    # −Δ of the 7-point stencil on the half spectrum of rfftn:
    # Σ_a 4/h² sin²(π k_a / n) with the integer wave numbers k_a
    k = np.fft.fftfreq(n, 1.0 / n)
    k_last = np.fft.rfftfreq(n, 1.0 / n)
    symbol = [4.0 / h ** 2 * np.sin(np.pi * kk / n) ** 2
              for kk in (k, k, k_last)]
    neg_laplace = xp.asarray(symbol[0][:, None, None]
                             + symbol[1][None, :, None]
                             + symbol[2][None, None, :])
    denominator = 1.0 + dt * eps ** 2 * neg_laplace ** 2

    c = xp.asarray(0.05 * np.random.default_rng(seed).standard_normal(
        (n, n, n)))
    for step in range(1, steps + 1):
        # implicit in the stiff 4th-order term, explicit in the nonlinearity
        c_hat = xp.fft.rfftn(c)
        c_hat -= dt * neg_laplace * xp.fft.rfftn(c * c * c - c)
        c_hat /= denominator
        c = xp.fft.irfftn(c_hat, s=(n, n, n), axes=(0, 1, 2))
        if log is not None and (step % max(steps // 10, 1) == 0
                                or step == steps):
            log(f"step {step}/{steps}: c in [{float(c.min()):+.3f}, "
                f"{float(c.max()):+.3f}], mean {float(c.mean()):+.5f}")
    return c.get() if xp is not np else c


def write_start(path, rho, description):
    """``rho`` (``(N, N, N)``) as a one-frame file in the study's layout
    (CDF-5, the format muGrid's PnetCDF reader opens)."""
    import netCDF4

    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as data:
        dims = ("frame", "tensor_dim__phase_field_solution-0", "nx", "ny",
                "nz")
        for name, size in zip(dims, (1, 1) + rho.shape):
            data.createDimension(name, size)
        variable = data.createVariable("phase_field_solution", "f8", dims)
        variable[0, 0] = rho
        data.description = description
        data.muFFTTO_layout_phase_field_solution = np.array([1, 1],
                                                            dtype=np.int32)


def plot_mid_slice(path, rho, title):
    """The mid slice ``z = 1/2`` and a zoom on its first quarter. Skipped
    without matplotlib (e.g. a cluster venv for the solves only)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"no matplotlib: {path} not written", flush=True)
        return

    n = rho.shape[0]
    mid = rho[:, :, n // 2]
    quarter = slice(0, max(n // 4, 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    for ax, image, label in ((axes[0], mid, "z = 1/2"),
                             (axes[1], mid[quarter, quarter],
                              "zoom: first quarter")):
        ax.imshow(image.T, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=max(n // 7, 100))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("nb_grid_pts", type=int, help="N of the N³ grid")
    parser.add_argument("output", help="NetCDF file of the start design")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--eta-h", type=float, default=2.0,
                        help="interface width of the optimisation in grid "
                             "spacings (large_3d.py: 2)")
    parser.add_argument("--device", default="auto",
                        choices=("auto", "gpu", "cpu"))
    args = parser.parse_args()

    started = time.time()
    n = args.nb_grid_pts
    xp = array_module(args.device)
    print(f"Cahn-Hilliard start {n}^3, seed {args.seed}, eta {args.eta_h:g} h,"
          f" eps = sqrt(2) eta, on {'the GPU' if xp is not np else 'the CPU'}",
          flush=True)
    c = cahn_hilliard(n, seed=args.seed, eta_h=args.eta_h, xp=xp,
                      log=lambda line: print(line, flush=True))
    rho = np.clip((1.0 - c) / 2.0, 0.0, 1.0)
    description = (f"Cahn-Hilliard start at {n}^3 from noise (seed "
                   f"{args.seed}), eps = sqrt(2) eta, eta = {args.eta_h:g} h,"
                   " T = 250 eps^2, rho = (1 - c) / 2 "
                   "(muTopOpt benchmarks/cahn_hilliard_start.py)")
    write_start(args.output, rho, description)
    grey = float(np.mean((rho > 0.1) & (rho < 0.9)))
    plot_mid_slice(args.output.rsplit(".", 1)[0] + ".png", rho,
                   f"Cahn-Hilliard start {n}³, seed {args.seed}: solid "
                   f"{np.mean(rho > 0.5):.3f}, grey {grey:.3f}")
    print(f"wrote {args.output}: mean rho {rho.mean():.4f}, solid fraction "
          f"(rho > 0.5) {np.mean(rho > 0.5):.4f}, grey fraction "
          f"(0.1 < rho < 0.9) {grey:.4f}, {time.time() - started:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
