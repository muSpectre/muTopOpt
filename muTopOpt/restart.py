#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Restarting from the NetCDF output of a previous run.

:func:`read_final_density` reads the last density frame of a file written by
``simulate.py`` -- all file I/O goes through muGrid's :class:`FileIONetCDF`
(the files are CDF-5/PnetCDF, which generic Python readers cannot parse).
:func:`fourier_resample` moves a periodic density between grids by
zero-padding/truncating its Fourier spectrum, so a converged coarse design can
seed a finer run (or vice versa) without smearing or shifting the interfaces.
:func:`restart_density` combines the two.

muGrid's Python API does not (yet) expose the grid shape or the scalar type of
a variable stored in a file, but both must be known *before* the field
collection is registered for reading -- registering a ``float64`` field for a
``float`` variable silently misreads the data in serial muGrid builds. The
grid and precision are therefore recovered from the file's global attributes:
the explicit ``nb_grid_pts``/``precision`` attributes when present (files
written by current ``simulate.py``), else parsed from the ``command_line``
attribute that every ``simulate.py`` run records.
"""

import shlex

import muGrid
import numpy as np

#: The initial-density kinds understood by :func:`muTopOpt.optimize.
#: initial_density`; any other ``--init`` value names a restart file.
INITIAL_DENSITY_KINDS = ("uniform", "random", "filtered_random")


def _resample_axis(a, n_new, axis):
    """Fourier-resample a real array along one axis (periodic band-limited
    interpolation): the spectrum is truncated (downsampling) or zero-padded
    (upsampling), with the usual Nyquist-bin handling that keeps the result
    real and makes up/down round trips of band-limited data exact."""
    n_old = a.shape[axis]
    if n_new == n_old:
        return a
    s = np.moveaxis(np.fft.fft(a, axis=axis), axis, -1)
    t = np.zeros(s.shape[:-1] + (int(n_new),), dtype=complex)
    m = min(n_old, n_new)
    npos = (m + 1) // 2  # non-negative frequencies 0 .. npos-1
    nneg = m // 2  # negative frequencies -nneg .. -1
    t[..., :npos] = s[..., :npos]
    if nneg:
        t[..., -nneg:] = s[..., -nneg:]
    if m % 2 == 0:
        # A Nyquist bin (frequency +-m/2) is involved. Real input makes the
        # old bin value real, so splitting/folding preserves realness.
        ny = m // 2
        if n_new < n_old:
            # Downsampling: the old +m/2 and -m/2 bins both alias onto the
            # single new Nyquist bin; -m/2 was copied above, add +m/2.
            t[..., ny] += s[..., ny]
        else:
            # Upsampling: the single old Nyquist bin (copied above to the
            # new -m/2 slot) splits into half-amplitude +-m/2 bins.
            t[..., -ny] *= 0.5
            t[..., ny] = t[..., -ny]
    out = np.fft.ifft(t).real * (n_new / n_old)
    return np.moveaxis(out, -1, axis)


def fourier_resample(a, nb_grid_pts):
    """Resample a periodic real field to a new grid by Fourier interpolation.

    Trigonometric (band-limited) interpolation: upsampling zero-pads the
    spectrum, downsampling truncates it. The mean is preserved exactly;
    interfaces stay in place (no half-pixel shifts). Note the result of
    downsampling -- or of upsampling a field with sharp interfaces (Gibbs
    ringing) -- can overshoot the range of the input; clip afterwards if the
    field must stay in ``[0, 1]``.

    Parameters
    ----------
    a : array_like
        Real field on the periodic source grid.
    nb_grid_pts : sequence of int
        Target grid shape (same dimensionality as ``a``).

    Returns
    -------
    np.ndarray
        The field resampled to ``nb_grid_pts`` (float64).
    """
    a = np.asarray(a, dtype=np.float64)
    if len(nb_grid_pts) != a.ndim:
        raise ValueError(
            f"target shape {tuple(nb_grid_pts)} does not match the "
            f"{a.ndim}-dimensional input field")
    for ax, n_new in enumerate(nb_grid_pts):
        a = _resample_axis(a, int(n_new), ax)
    return a


def _parse_option(tokens, names, nb_values):
    """Return the value(s) of a command-line option from a recorded ``argv``
    token list: the ``nb_values`` tokens following any of ``names`` (also
    accepting the ``--name=value`` form for single-valued options), or
    ``None`` if absent."""
    for i, tok in enumerate(tokens):
        if tok in names:
            values = tokens[i + 1:i + 1 + nb_values]
            if len(values) == nb_values:
                return values
        for name in names:
            if nb_values == 1 and tok.startswith(name + "="):
                return [tok[len(name) + 1:]]
    return None


def _file_metadata(fio):
    """Recover grid shape, precision and domain lengths of the stored density
    from the file's global attributes (explicit attributes preferred, the
    recorded ``command_line`` as fallback for files written before they were
    introduced)."""
    names = fio.read_global_attribute_names()
    tokens = (shlex.split(str(fio.read_global_attribute("command_line")))
              if "command_line" in names else [])

    if "nb_grid_pts" in names:
        nb_grid_pts = [int(n) for n in fio.read_global_attribute("nb_grid_pts")]
    else:
        # -n takes 2 or 3 values; take up to 3 and drop trailing non-integers
        # (the value count is not recorded, only the tokens).
        values = _parse_option(tokens, ("-n", "--nb-grid-pts"), 3)
        if values is None:
            values = _parse_option(tokens, ("-n", "--nb-grid-pts"), 2)
        if values is None:
            raise ValueError(
                "cannot determine the grid of the previous run: the file has "
                "neither an 'nb_grid_pts' global attribute nor a "
                "'command_line' attribute with -n/--nb-grid-pts")
        nb_grid_pts = []
        for v in values:
            try:
                nb_grid_pts.append(int(v))
            except ValueError:
                break
        if len(nb_grid_pts) not in (2, 3):
            raise ValueError(
                f"cannot parse -n/--nb-grid-pts from the recorded command "
                f"line {' '.join(tokens)!r}")

    if "precision" in names:
        precision = str(fio.read_global_attribute("precision")).strip()
    else:
        values = _parse_option(tokens, ("--precision",), 1)
        precision = values[0] if values is not None else "double"
    if precision not in ("single", "double"):
        raise ValueError(f"unknown precision {precision!r} in restart file")

    domain_lengths = (
        [float(x) for x in fio.read_global_attribute("domain_lengths")]
        if "domain_lengths" in names else None)

    return {
        "nb_grid_pts": tuple(nb_grid_pts),
        "precision": precision,
        "domain_lengths": domain_lengths,
    }


def read_final_density(filename):
    """Read the last density frame of a previous run's NetCDF output.

    The full (global) field is read through muGrid with a *serial*
    communicator, so under MPI every rank independently reads the whole
    (old) grid -- restart initialization is a one-off, and the resampling
    below needs the global field anyway.

    Parameters
    ----------
    filename : str
        NetCDF output of a previous ``simulate.py`` run.

    Returns
    -------
    rho : np.ndarray
        The density of the last frame on the file's global grid (float64).
    metadata : dict
        ``nb_grid_pts`` (tuple), ``precision`` (str), ``domain_lengths``
        (list or None) and ``frame`` (index of the frame read) of the file.
    """
    fio = muGrid.FileIONetCDF(
        filename, muGrid.FileIONetCDF.OpenMode.Read, muGrid.Communicator()
    )
    try:
        metadata = _file_metadata(fio)
        nb_frames = len(fio)
        if nb_frames < 1:
            raise ValueError(f"restart file '{filename}' contains no frames")
        # The field must be registered with the dtype the variable was
        # *stored* with: muGrid reads raw (untyped) in serial builds, so a
        # dtype mismatch would silently misinterpret the bytes.
        dtype = np.float32 if metadata["precision"] == "single" else np.float64
        fc = muGrid.GlobalFieldCollection(list(metadata["nb_grid_pts"]))
        field = fc.real_field("density", dtype=dtype)
        fio.register_field_collection(fc, field_names=["density"])
        fio.read(nb_frames - 1, ["density"])
        rho = np.array(field.p, dtype=np.float64)  # copy out of the collection
        metadata["frame"] = nb_frames - 1
    finally:
        fio.close()
    return rho, metadata


def restart_density(filename, nb_grid_pts):
    """Initial density from a previous run: the last frame of ``filename``,
    Fourier-resampled to ``nb_grid_pts`` if the grids differ and clipped back
    to the admissible range ``[0, 1]`` (resampling a sharp design rings and
    can overshoot the bounds).

    Returns ``(rho, metadata)`` with ``rho`` the global density on the target
    grid (float64) and ``metadata`` as in :func:`read_final_density`, plus
    ``resampled`` (bool).
    """
    rho, metadata = read_final_density(filename)
    nb_grid_pts = tuple(int(n) for n in nb_grid_pts)
    metadata["resampled"] = nb_grid_pts != metadata["nb_grid_pts"]
    if metadata["resampled"]:
        rho = fourier_resample(rho, nb_grid_pts)
    return np.clip(rho, 0.0, 1.0), metadata
