"""``plot_3d()`` — interactive 3D geometry as a plotly figure (design §3, §6).

Structures become ``Mesh3d`` (boxes -> 8 verts / 12 tris; spheres ->
parametric mesh), sources ``Scatter3d`` markers / translucent planes, monitors
translucent boxes / planes, plus a wireframe domain box and translucent PML
shells on non-periodic boundaries.

plotly is the optional ``photonhub[viz]`` extra and is imported LAZILY: when it
is absent, :func:`plot_3d` raises ``ImportError`` with the exact
``pip install photonhub[viz]`` hint (design §8).
"""

import math

from . import _geometry as geom
from . import _style

_PLOTLY_HINT = (
    "plot_3d requires plotly (the optional 3D viz extra). Install it with:\n"
    "    pip install photonhub[viz]"
)


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without plotly
        raise ImportError(_PLOTLY_HINT) from exc
    return go


def _eps_color(permittivity, vmin, vmax):
    r, g, b, _ = _style.eps_facecolor(permittivity, vmin, vmax)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def _box_mesh(go, center, size, color, name, opacity=1.0):
    cx, cy, cz = center
    hx, hy, hz = (s / 2.0 for s in size)
    xs = [cx - hx, cx + hx, cx + hx, cx - hx, cx - hx, cx + hx, cx + hx, cx - hx]
    ys = [cy - hy, cy - hy, cy + hy, cy + hy, cy - hy, cy - hy, cy + hy, cy + hy]
    zs = [cz - hz, cz - hz, cz - hz, cz - hz, cz + hz, cz + hz, cz + hz, cz + hz]
    # 12 triangles (two per face), standard unit-cube triangulation.
    i = [0, 0, 0, 0, 4, 4, 6, 6, 1, 1, 2, 2]
    j = [1, 2, 4, 3, 5, 6, 5, 7, 5, 6, 6, 7]
    k = [2, 3, 5, 7, 6, 7, 1, 3, 6, 2, 7, 3]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color,
                     opacity=opacity, name=name, showscale=False,
                     flatshading=True)


def _sphere_mesh(go, center, radius, color, name, n=16):
    cx, cy, cz = center
    us = [math.pi * a / n for a in range(n + 1)]          # polar [0, pi]
    vs = [2 * math.pi * b / n for b in range(n + 1)]      # azimuth [0, 2pi]
    xs, ys, zs = [], [], []
    for u in us:
        for v in vs:
            xs.append(cx + radius * math.sin(u) * math.cos(v))
            ys.append(cy + radius * math.sin(u) * math.sin(v))
            zs.append(cz + radius * math.cos(u))
    return go.Mesh3d(x=xs, y=ys, z=zs, alphahull=0, color=color, opacity=1.0,
                     name=name, showscale=False, flatshading=True)


# Axis-letter -> (transverse u index, transverse v index, axial index). Mirrors
# the (u, v) transverse convention in viz/_geometry.in_plane_axes.
_AXIS_FRAME = {
    "x": (1, 2, 0),  # u=y, v=z, axial=x
    "y": (0, 2, 1),  # u=x, v=z, axial=y
    "z": (0, 1, 2),  # u=x, v=y, axial=z
}


def _embed_uv(u, v, w, u_i, v_i, a_i):
    """Place a transverse ``(u, v)`` point at axial coordinate ``w`` into a 3D
    ``(x, y, z)`` tuple using the axis frame ``(u_i, v_i, a_i)``."""
    p = [0.0, 0.0, 0.0]
    p[u_i] = u
    p[v_i] = v
    p[a_i] = w
    return p[0], p[1], p[2]


def _prism_mesh(go, axis, vertices_uv, axial_lo, axial_hi, color, name,
                opacity=1.0):
    """A polygon (``vertices_uv`` in transverse (u, v)) extruded along ``axis``
    between ``axial_lo`` and ``axial_hi`` as a closed ``Mesh3d`` prism.

    Bottom + top caps are fan-triangulated (valid for convex polygons; concave
    polygons still render a watertight wall and a possibly over-covered cap —
    acceptable for a schematic). Sidewall slant is approximated as vertical
    (``sidewall_angle`` ignored), documented at the call site."""
    n = len(vertices_uv)
    u_i, v_i, a_i = _AXIS_FRAME[axis]
    xs, ys, zs = [], [], []
    for (u, v) in vertices_uv:                       # bottom ring [0, n)
        x, y, z = _embed_uv(u, v, axial_lo, u_i, v_i, a_i)
        xs.append(x); ys.append(y); zs.append(z)
    for (u, v) in vertices_uv:                       # top ring [n, 2n)
        x, y, z = _embed_uv(u, v, axial_hi, u_i, v_i, a_i)
        xs.append(x); ys.append(y); zs.append(z)
    i, j, k = [], [], []
    # Caps: triangle fan from vertex 0 of each ring.
    for t in range(1, n - 1):
        i.append(0); j.append(t); k.append(t + 1)              # bottom
        i.append(n); j.append(n + t + 1); k.append(n + t)      # top (flipped)
    # Walls: two triangles per edge connecting bottom ring to top ring.
    for e in range(n):
        b0, b1 = e, (e + 1) % n
        t0, t1 = n + e, n + (e + 1) % n
        i.append(b0); j.append(b1); k.append(t1)
        i.append(b0); j.append(t1); k.append(t0)
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color,
                     opacity=opacity, name=name, showscale=False,
                     flatshading=True)


def _polyslab_mesh(go, g, color, name):
    """PolySlab -> extruded-polygon prism. ``slab_bounds_um`` are the axial
    (lo, hi); the polygon lives in the two transverse axes. Vertical walls
    (sidewall_angle approximated as 0)."""
    lo, hi = g.slab_bounds_um
    return _prism_mesh(go, g.axis, g.vertices_um, lo, hi, color, name)


def _cylinder_mesh(go, g, color, name, n=48):
    """Cylinder (solid disk / ring / annular sector) -> a parametric tube
    ``Mesh3d``. The arc is sampled into an (annular) polygon cross-section that
    is then extruded along ``axis`` by :func:`_prism_mesh`. A solid disk
    (``inner_radius_um == 0``) reduces to the outer-arc polygon closed through
    the centre; the curved wall is faceted into ``n`` segments per full turn."""
    a_i = "xyz".index(g.axis)
    u_i, v_i, _ = _AXIS_FRAME[g.axis]
    cu = g.center_um[u_i]
    cv = g.center_um[v_i]
    half = g.length_um / 2.0
    lo = g.center_um[a_i] - half
    hi = g.center_um[a_i] + half
    verts = geom._arc_polygon(cu, cv, g.radius_um, g.inner_radius_um,
                              g.angle_start, g.angle_stop, n=n)
    return _prism_mesh(go, g.axis, verts, lo, hi, color, name)


def _domain_wireframe(go, size):
    """12 edges of the domain box [0,Lx] x [0,Ly] x [0,Lz] as one Scatter3d."""
    Lx, Ly, Lz = size
    c = [(0, 0, 0), (Lx, 0, 0), (Lx, Ly, 0), (0, Ly, 0),
         (0, 0, Lz), (Lx, 0, Lz), (Lx, Ly, Lz), (0, Ly, Lz)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [c[a][0], c[b][0], None]
        ys += [c[a][1], c[b][1], None]
        zs += [c[a][2], c[b][2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color="#444", width=2), name="domain",
                        showlegend=True)


def _pml_shells(go, sim):
    """Translucent slab shells on each non-periodic (PML) boundary face."""
    from ._style import _axis_spacings_um

    shells = []
    realized = sim._realized_um()
    layers = sim.pml_num_layers
    for axis_i, letter in enumerate("xyz"):
        if getattr(sim.boundaries, letter) != "pml":
            continue
        lo_dl, hi_dl, length = _axis_spacings_um(sim, axis_i)
        lo_t = layers * lo_dl
        hi_t = layers * hi_dl
        for face_lo, face_hi, tag in (
            (0.0, lo_t, "low"), (length - hi_t, length, "high"),
        ):
            size = list(realized)
            center = [r / 2.0 for r in realized]
            size[axis_i] = face_hi - face_lo
            center[axis_i] = 0.5 * (face_lo + face_hi)
            shells.append(_box_mesh(go, center, size, _style.PML_COLOR,
                                   f"PML {letter}-{tag}", opacity=0.12))
    return shells


def plot_3d(sim, **kw):
    """Build and return a plotly ``Figure`` of the 3D scene geometry. Raises
    ``ImportError`` (with the install hint) when plotly is not installed."""
    go = _require_plotly()
    fig = go.Figure()

    eps_vals = ([sim.background.permittivity]
                + [s.medium.permittivity for s in sim.structures])
    vmin, vmax = _style.eps_norm(eps_vals)

    # Structures.
    for n, structure in enumerate(sim.structures):
        g = structure.geometry
        color = _eps_color(structure.medium.permittivity, vmin, vmax)
        gtype = getattr(g, "type", None)
        if gtype == "box":
            fig.add_trace(_box_mesh(go, g.center_um, g.size_um, color,
                                   f"structure{n}"))
        elif gtype == "sphere":
            fig.add_trace(_sphere_mesh(go, g.center_um, g.radius_um, color,
                                      f"structure{n}"))
        elif gtype == "polyslab":
            fig.add_trace(_polyslab_mesh(go, g, color, f"structure{n}"))
        elif gtype == "cylinder":
            fig.add_trace(_cylinder_mesh(go, g, color, f"structure{n}"))

    # Sources.
    realized = sim._realized_um()
    for n, s in enumerate(sim.sources):
        stype = getattr(s, "type", None)
        if stype == "point_dipole":
            cx, cy, cz = s.center_um
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz], mode="markers",
                marker=dict(size=6, color=_style.SOURCE_COLOR),
                name=f"source{n}"))
        elif stype == "plane_wave":
            fig.add_trace(_plane_mesh(go, s.axis, s.position_um, realized,
                                     _style.SOURCE_COLOR, f"source{n}"))

    # Monitors.
    for n, m in enumerate(sim.monitors):
        mtype = getattr(m, "type", None)
        if mtype == "field_dft":
            size = [max(d, 1e-6) for d in m.size_um]  # thin box for 0-extent
            fig.add_trace(_box_mesh(go, m.center_um, size, _style.MONITOR_COLOR,
                                   f"monitor:{m.name}", opacity=0.2))
        elif mtype == "flux":
            fig.add_trace(_plane_mesh(go, m.axis, m.position_um, realized,
                                     _style.MONITOR_COLOR, f"monitor:{m.name}"))

    # Domain wireframe + PML shells.
    fig.add_trace(_domain_wireframe(go, realized))
    for shell in _pml_shells(go, sim):
        fig.add_trace(shell)

    fig.update_layout(
        scene=dict(
            xaxis_title="x (µm)", yaxis_title="y (µm)", zaxis_title="z (µm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="3D scene",
    )
    return fig


def _plane_mesh(go, axis, position, realized, color, name):
    """A translucent full-span plane perpendicular to ``axis`` at ``position``
    (PlaneWave source / FluxMonitor)."""
    ai = "xyz".index(axis)
    size = [r for r in realized]
    center = [r / 2.0 for r in realized]
    size[ai] = max(realized[ai] * 1e-3, 1e-6)  # near-zero thickness
    center[ai] = position
    return _box_mesh(go, center, size, color, name, opacity=0.2)
