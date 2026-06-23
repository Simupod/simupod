"""``plot_mode()`` — the transverse field of an FDE eigenmode as a heatmap.

Renders a :class:`photonhub.plugins.modes.Mode` (the dominant transverse
component ``Ex``/``Ey`` of a guided mode) on its real-space µm cross-section.
The field is signed, so it uses the same diverging, zero-centered colormap the
field views use for a real/imag slice (design §7). The title carries the
component, the modal index ``n_eff``, and the free-space wavelength.

Consumes the mode's :meth:`~photonhub.plugins.modes.Mode.field_dataarray`
(an :class:`xarray.DataArray` already in µm x/y coords with ``n_eff`` /
``polarization`` / ``wavelength_um`` attrs), so it stays decoupled from the
solver internals. Returns the matplotlib ``Axes``; never calls ``plt.show()``.
"""

import numpy as np

from . import _style


def plot_mode(mode, *, ax=None, cmap=None, legend=False, **kw):
    """Heatmap of an FDE :class:`~photonhub.plugins.modes.Mode`'s transverse
    field on its µm cross-section. Returns the matplotlib ``Axes``.

    ``mode`` is a ``Mode`` (carrying ``.field``, ``.n_eff``, ``.polarization``,
    ``.wavelength_um`` and ``.field_dataarray()``). ``cmap=`` overrides the
    diverging colormap (the symmetric zero-centered normalization is kept).
    ``ax=`` draws into an existing Axes. Extra ``**kw`` pass through to
    ``pcolormesh``."""
    import matplotlib.pyplot as plt

    da = mode.field_dataarray()
    arr = np.asarray(da.values, dtype=np.float64)
    xs = np.asarray(da.coords["x"].values, dtype=np.float64)
    ys = np.asarray(da.coords["y"].values, dtype=np.float64)
    component = str(da.attrs.get("component", da.name or "E"))

    if ax is None:
        _, ax = plt.subplots()

    # Signed field -> the §7 diverging map, symmetric about 0 (reuse the field
    # colormap/normalization selection so a mode and a field slice match).
    cmap_name, norm = _style.field_cmap_and_norm(component, "real", arr, cmap)
    mesh = ax.pcolormesh(xs, ys, arr, cmap=cmap_name, norm=norm,
                         shading="nearest", **kw)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(f"{component} (mode field)")

    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(
        f"{component} mode  (n_eff={float(mode.n_eff):.4g}, "
        f"λ={float(mode.wavelength_um):.4g} µm)"
    )
    return ax
